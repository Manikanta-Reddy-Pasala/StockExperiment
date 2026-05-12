# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 882.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 137 |
| ALERT1 | 93 |
| ALERT2 | 92 |
| ALERT2_SKIP | 39 |
| ALERT3 | 243 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 84 |
| PARTIAL | 12 |
| TARGET_HIT | 8 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 42 / 54
- **Target hits / Stop hits / Partials:** 8 / 76 / 12
- **Avg / median % per leg:** 0.86% / -0.45%
- **Sum % (uncompounded):** 82.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 19 | 38.0% | 6 | 43 | 1 | 0.39% | 19.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.36% | 6.7% |
| BUY @ 3rd Alert (retest2) | 48 | 17 | 35.4% | 6 | 42 | 0 | 0.27% | 12.8% |
| SELL (all) | 46 | 23 | 50.0% | 2 | 33 | 11 | 1.38% | 63.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 23 | 50.0% | 2 | 33 | 11 | 1.38% | 63.4% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.36% | 6.7% |
| retest2 (combined) | 94 | 40 | 42.6% | 8 | 75 | 11 | 0.81% | 76.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 565.45 | 558.52 | 558.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 13:15:00 | 579.59 | 573.60 | 569.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 576.86 | 577.36 | 572.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 10:00:00 | 576.86 | 577.36 | 572.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 584.25 | 589.58 | 583.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 582.61 | 589.58 | 583.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 583.50 | 588.36 | 583.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 583.03 | 588.36 | 583.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 578.23 | 586.34 | 582.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 578.23 | 586.34 | 582.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 572.25 | 583.52 | 581.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:30:00 | 570.40 | 583.52 | 581.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 570.75 | 578.76 | 579.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 568.23 | 575.97 | 578.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 562.75 | 562.48 | 568.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 12:00:00 | 562.75 | 562.48 | 568.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 567.03 | 564.19 | 567.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 567.03 | 564.19 | 567.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 566.23 | 564.59 | 567.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 567.67 | 564.59 | 567.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 561.76 | 564.03 | 566.86 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 567.98 | 566.15 | 566.12 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 15:15:00 | 565.50 | 566.04 | 566.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 558.25 | 564.48 | 565.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 562.50 | 561.90 | 563.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 562.50 | 561.90 | 563.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 562.50 | 561.90 | 563.28 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 570.49 | 564.93 | 564.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 09:15:00 | 575.50 | 568.47 | 566.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 13:15:00 | 570.36 | 570.39 | 568.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-30 14:00:00 | 570.36 | 570.39 | 568.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 568.81 | 570.08 | 568.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:45:00 | 565.01 | 570.08 | 568.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 565.75 | 569.21 | 567.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 572.00 | 569.21 | 567.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 560.65 | 567.16 | 567.14 | SL hit (close<static) qty=1.00 sl=565.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 561.25 | 565.97 | 566.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 12:15:00 | 558.20 | 564.42 | 565.84 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 591.04 | 566.68 | 566.09 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 532.16 | 573.11 | 573.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 511.99 | 560.88 | 567.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 546.75 | 541.54 | 552.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 546.75 | 541.54 | 552.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 548.45 | 544.22 | 552.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:30:00 | 545.63 | 544.22 | 552.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 555.45 | 546.47 | 552.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 555.45 | 546.47 | 552.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 559.50 | 549.07 | 553.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 559.50 | 549.07 | 553.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 581.41 | 557.81 | 556.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 588.89 | 573.51 | 566.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 13:15:00 | 665.05 | 665.92 | 651.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 14:00:00 | 665.05 | 665.92 | 651.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 658.05 | 664.53 | 658.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 658.05 | 664.53 | 658.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 659.00 | 663.42 | 658.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 655.00 | 663.42 | 658.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 658.00 | 662.34 | 658.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:30:00 | 665.70 | 660.61 | 658.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 671.35 | 663.81 | 661.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 663.00 | 684.97 | 681.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 667.95 | 678.18 | 678.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 667.95 | 678.18 | 678.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 649.50 | 669.54 | 674.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 636.00 | 626.41 | 635.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 636.00 | 626.41 | 635.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 636.00 | 626.41 | 635.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 642.95 | 626.41 | 635.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 630.80 | 627.29 | 634.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:15:00 | 623.55 | 627.63 | 633.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 624.70 | 627.50 | 631.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:15:00 | 592.37 | 611.07 | 620.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:15:00 | 593.47 | 611.07 | 620.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-03 13:15:00 | 561.19 | 569.96 | 581.50 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 549.25 | 536.48 | 535.02 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 529.50 | 540.21 | 540.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 528.45 | 531.81 | 533.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 12:15:00 | 537.00 | 532.34 | 533.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 537.00 | 532.34 | 533.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 537.00 | 532.34 | 533.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:45:00 | 537.85 | 532.34 | 533.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 543.05 | 534.48 | 534.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 14:15:00 | 546.00 | 536.79 | 535.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 11:15:00 | 649.45 | 652.32 | 637.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 12:00:00 | 649.45 | 652.32 | 637.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 648.80 | 653.02 | 643.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:45:00 | 665.70 | 653.81 | 646.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 603.45 | 643.18 | 644.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 603.45 | 643.18 | 644.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 595.00 | 613.68 | 625.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 605.55 | 605.13 | 616.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:30:00 | 605.60 | 605.13 | 616.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 613.90 | 607.82 | 615.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 608.85 | 607.82 | 615.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 609.55 | 608.16 | 615.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 613.75 | 608.16 | 615.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 604.20 | 599.79 | 605.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 608.25 | 599.79 | 605.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 602.75 | 600.38 | 605.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:45:00 | 604.85 | 600.38 | 605.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 603.85 | 601.07 | 604.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:30:00 | 604.25 | 601.07 | 604.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 603.55 | 601.57 | 604.82 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 608.80 | 605.91 | 605.91 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 601.00 | 604.94 | 605.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 596.00 | 600.93 | 603.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 09:15:00 | 600.00 | 596.76 | 600.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 600.00 | 596.76 | 600.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 600.00 | 596.76 | 600.26 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 613.00 | 603.31 | 602.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 625.00 | 609.04 | 605.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 692.90 | 697.18 | 685.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:15:00 | 735.55 | 696.07 | 689.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 10:15:00 | 772.33 | 717.06 | 700.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-08-27 10:15:00 | 748.25 | 749.38 | 728.56 | SL hit (close<ema200) qty=0.50 sl=749.38 alert=retest1 |

### Cycle 18 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 725.90 | 738.13 | 738.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 10:15:00 | 712.55 | 726.58 | 732.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 13:15:00 | 728.70 | 723.72 | 729.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 14:00:00 | 728.70 | 723.72 | 729.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 717.25 | 722.43 | 727.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:45:00 | 729.50 | 722.43 | 727.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 733.00 | 724.95 | 728.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 754.55 | 724.95 | 728.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 740.35 | 728.03 | 729.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 740.35 | 728.03 | 729.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 12:15:00 | 742.30 | 732.16 | 731.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 14:15:00 | 754.00 | 737.69 | 733.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 742.05 | 751.51 | 745.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 742.05 | 751.51 | 745.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 742.05 | 751.51 | 745.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:15:00 | 738.50 | 751.51 | 745.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 753.15 | 751.84 | 746.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:30:00 | 737.75 | 751.84 | 746.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 758.60 | 760.18 | 754.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:30:00 | 756.90 | 760.18 | 754.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 771.05 | 762.36 | 755.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 13:45:00 | 773.05 | 763.77 | 757.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 15:00:00 | 773.75 | 765.76 | 759.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 729.35 | 757.85 | 759.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 729.35 | 757.85 | 759.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 723.90 | 742.83 | 751.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 738.85 | 736.95 | 745.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 738.85 | 736.95 | 745.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 738.85 | 736.95 | 745.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 743.15 | 736.95 | 745.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 743.00 | 739.27 | 745.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:00:00 | 743.00 | 739.27 | 745.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 739.35 | 739.29 | 744.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:15:00 | 732.35 | 739.29 | 744.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 748.10 | 741.13 | 743.31 | SL hit (close>static) qty=1.00 sl=744.65 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 13:15:00 | 748.05 | 744.49 | 744.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 14:15:00 | 752.15 | 746.02 | 745.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 741.40 | 745.90 | 745.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 741.40 | 745.90 | 745.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 741.40 | 745.90 | 745.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 741.40 | 745.90 | 745.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 746.00 | 745.92 | 745.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:15:00 | 741.90 | 745.92 | 745.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 744.80 | 745.69 | 745.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 746.35 | 745.69 | 745.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 741.55 | 744.86 | 744.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 13:15:00 | 739.80 | 743.85 | 744.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 745.50 | 743.72 | 744.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 745.50 | 743.72 | 744.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 745.50 | 743.72 | 744.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 752.90 | 743.72 | 744.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 753.60 | 745.70 | 745.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 14:15:00 | 756.95 | 751.15 | 748.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 13:15:00 | 755.40 | 757.49 | 753.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:00:00 | 755.40 | 757.49 | 753.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 766.35 | 761.51 | 757.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:30:00 | 756.25 | 761.51 | 757.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 775.40 | 780.22 | 772.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 775.40 | 780.22 | 772.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 746.25 | 772.43 | 770.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 746.25 | 772.43 | 770.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 742.45 | 766.44 | 767.56 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 790.05 | 766.64 | 765.18 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 766.75 | 774.87 | 775.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 761.00 | 766.03 | 770.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 761.75 | 758.93 | 763.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 761.75 | 758.93 | 763.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 761.75 | 758.93 | 763.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 761.75 | 758.93 | 763.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 749.75 | 752.95 | 756.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:45:00 | 754.65 | 752.95 | 756.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 762.00 | 754.76 | 757.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 762.00 | 754.76 | 757.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 760.00 | 755.81 | 757.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 735.85 | 755.81 | 757.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 699.06 | 729.31 | 739.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 724.65 | 724.55 | 733.41 | SL hit (close>ema200) qty=0.50 sl=724.55 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 740.25 | 726.27 | 725.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 13:15:00 | 748.35 | 730.68 | 727.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 772.90 | 777.48 | 765.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 772.90 | 777.48 | 765.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 938.55 | 937.76 | 921.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:30:00 | 951.40 | 944.96 | 926.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-18 14:15:00 | 1046.54 | 985.54 | 953.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 933.60 | 956.46 | 959.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 924.65 | 947.26 | 954.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 954.50 | 945.18 | 952.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 954.50 | 945.18 | 952.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 954.50 | 945.18 | 952.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 954.50 | 945.18 | 952.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 932.05 | 942.55 | 950.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 910.55 | 930.35 | 938.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 865.02 | 919.43 | 933.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 923.15 | 894.13 | 910.78 | SL hit (close>ema200) qty=0.50 sl=894.13 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 929.00 | 919.79 | 918.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 947.20 | 925.27 | 921.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 935.55 | 939.11 | 931.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 09:30:00 | 948.20 | 939.11 | 931.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 925.90 | 936.47 | 931.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 925.90 | 936.47 | 931.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 925.10 | 934.19 | 930.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:00:00 | 925.10 | 934.19 | 930.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 923.80 | 932.11 | 930.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:30:00 | 924.40 | 932.11 | 930.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 922.45 | 928.23 | 928.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 916.65 | 925.02 | 927.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 935.75 | 924.01 | 925.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 14:15:00 | 935.75 | 924.01 | 925.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 935.75 | 924.01 | 925.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 935.75 | 924.01 | 925.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 943.90 | 927.99 | 926.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 957.55 | 933.90 | 929.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 937.00 | 941.42 | 934.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 937.00 | 941.42 | 934.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 937.00 | 941.42 | 934.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 937.00 | 941.42 | 934.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 935.00 | 940.13 | 934.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 935.00 | 940.13 | 934.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 939.30 | 939.97 | 934.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:15:00 | 947.50 | 941.43 | 936.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 950.45 | 941.78 | 937.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 926.15 | 938.65 | 936.81 | SL hit (close<static) qty=1.00 sl=934.10 alert=retest2 |

### Cycle 32 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 919.35 | 934.79 | 935.22 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 944.65 | 934.49 | 934.42 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 932.00 | 936.30 | 936.88 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 09:15:00 | 1002.60 | 949.56 | 942.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 10:15:00 | 1019.90 | 963.63 | 949.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 10:15:00 | 986.05 | 987.47 | 971.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 986.05 | 987.47 | 971.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 972.45 | 982.82 | 973.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:45:00 | 974.95 | 982.82 | 973.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 973.20 | 980.90 | 973.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 15:15:00 | 974.30 | 980.90 | 973.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 974.30 | 979.58 | 973.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 987.80 | 979.58 | 973.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 984.85 | 980.63 | 974.35 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 968.05 | 971.90 | 972.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 915.60 | 960.64 | 967.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 925.60 | 923.07 | 940.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 925.60 | 923.07 | 940.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 915.10 | 899.02 | 909.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 915.10 | 899.02 | 909.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 910.95 | 901.41 | 909.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 904.30 | 906.54 | 910.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 12:15:00 | 910.80 | 902.54 | 901.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 910.80 | 902.54 | 901.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 925.95 | 909.56 | 905.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 908.50 | 913.67 | 909.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 908.50 | 913.67 | 909.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 908.50 | 913.67 | 909.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 908.50 | 913.67 | 909.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 903.40 | 911.61 | 908.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:30:00 | 901.85 | 908.02 | 907.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 901.50 | 906.72 | 906.99 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 12:15:00 | 910.30 | 907.14 | 907.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 913.50 | 908.56 | 907.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 10:15:00 | 906.40 | 909.53 | 908.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 10:15:00 | 906.40 | 909.53 | 908.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 906.40 | 909.53 | 908.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 905.45 | 909.53 | 908.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 905.00 | 908.62 | 908.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 906.10 | 908.62 | 908.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 912.90 | 909.48 | 908.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 926.65 | 909.83 | 909.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 901.75 | 907.84 | 908.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 901.75 | 907.84 | 908.56 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 914.95 | 907.81 | 907.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 12:15:00 | 946.50 | 920.45 | 914.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 14:15:00 | 953.10 | 957.18 | 948.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 15:00:00 | 953.10 | 957.18 | 948.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 964.15 | 971.65 | 962.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 964.15 | 971.65 | 962.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 966.05 | 970.53 | 963.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:00:00 | 975.05 | 970.92 | 964.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 15:00:00 | 973.80 | 970.65 | 965.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 999.15 | 970.58 | 965.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 11:15:00 | 986.50 | 996.35 | 996.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 11:15:00 | 986.50 | 996.35 | 996.91 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 996.70 | 994.97 | 994.96 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 991.85 | 994.35 | 994.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 987.45 | 992.43 | 993.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 967.20 | 960.41 | 970.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 11:15:00 | 967.20 | 960.41 | 970.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 967.20 | 960.41 | 970.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:00:00 | 967.20 | 960.41 | 970.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 964.00 | 961.13 | 969.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:15:00 | 954.00 | 961.48 | 969.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 12:15:00 | 906.30 | 914.88 | 923.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 15:15:00 | 920.30 | 915.61 | 921.92 | SL hit (close>ema200) qty=0.50 sl=915.61 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 970.05 | 921.08 | 919.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 980.50 | 968.78 | 955.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 982.10 | 982.11 | 972.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 982.10 | 982.11 | 972.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 970.90 | 978.93 | 972.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 970.90 | 978.93 | 972.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 972.00 | 977.54 | 972.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 963.00 | 977.54 | 972.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 961.85 | 974.41 | 971.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:15:00 | 959.35 | 974.41 | 971.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 941.15 | 967.75 | 968.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 937.30 | 958.06 | 964.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 950.00 | 943.81 | 953.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 950.00 | 943.81 | 953.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 948.70 | 944.78 | 952.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 949.90 | 944.78 | 952.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 955.95 | 947.02 | 952.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 955.95 | 947.02 | 952.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 965.90 | 950.79 | 954.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 965.90 | 950.79 | 954.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 971.00 | 957.33 | 956.70 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 942.35 | 955.07 | 955.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 936.80 | 951.42 | 954.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 852.30 | 848.17 | 872.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 852.30 | 848.17 | 872.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 852.30 | 848.17 | 872.60 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 904.00 | 881.06 | 878.03 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 13:15:00 | 862.35 | 874.59 | 875.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 10:15:00 | 843.60 | 863.79 | 869.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 804.00 | 802.19 | 818.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 10:30:00 | 804.10 | 802.19 | 818.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 810.15 | 803.79 | 817.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:45:00 | 810.90 | 803.79 | 817.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 782.95 | 801.87 | 811.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 776.65 | 801.87 | 811.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 737.82 | 755.14 | 779.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 746.35 | 732.67 | 752.60 | SL hit (close>ema200) qty=0.50 sl=732.67 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 659.65 | 651.66 | 651.41 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 621.80 | 647.02 | 649.43 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 701.80 | 648.52 | 642.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 711.70 | 661.16 | 649.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 678.10 | 692.60 | 674.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:00:00 | 678.10 | 692.60 | 674.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 677.00 | 689.48 | 674.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:30:00 | 673.50 | 689.48 | 674.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 676.75 | 686.94 | 675.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:15:00 | 674.35 | 686.94 | 675.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 676.55 | 684.86 | 675.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:30:00 | 673.40 | 684.86 | 675.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 676.00 | 681.68 | 675.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 674.60 | 681.68 | 675.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 680.50 | 681.44 | 675.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 692.00 | 681.44 | 675.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 688.50 | 681.32 | 676.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 669.60 | 678.03 | 676.31 | SL hit (close<static) qty=1.00 sl=673.75 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 657.25 | 672.75 | 674.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 653.00 | 668.80 | 672.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 633.75 | 631.47 | 644.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 633.75 | 631.47 | 644.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 633.75 | 631.47 | 644.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 628.80 | 631.47 | 644.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 635.05 | 632.19 | 643.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 643.45 | 632.19 | 643.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 642.75 | 634.30 | 643.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 641.50 | 634.30 | 643.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 647.90 | 637.02 | 643.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:45:00 | 650.05 | 637.02 | 643.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 656.35 | 640.89 | 644.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 656.35 | 640.89 | 644.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 651.15 | 646.26 | 646.55 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 648.05 | 646.93 | 646.82 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 644.40 | 646.77 | 646.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 628.85 | 642.90 | 645.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 15:15:00 | 617.00 | 615.77 | 623.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 09:15:00 | 607.90 | 615.77 | 623.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 613.40 | 613.70 | 619.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:45:00 | 621.15 | 613.70 | 619.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 614.95 | 613.95 | 618.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:45:00 | 614.60 | 613.95 | 618.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 624.45 | 615.93 | 618.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:30:00 | 626.55 | 615.93 | 618.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 619.30 | 616.60 | 618.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 624.55 | 616.60 | 618.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 619.50 | 617.18 | 619.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:00:00 | 616.85 | 617.12 | 618.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:45:00 | 615.00 | 615.35 | 617.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 625.15 | 617.31 | 618.04 | SL hit (close>static) qty=1.00 sl=621.90 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 12:15:00 | 628.55 | 620.43 | 619.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 641.05 | 624.56 | 621.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 623.50 | 627.52 | 623.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 623.50 | 627.52 | 623.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 623.50 | 627.52 | 623.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 623.50 | 627.52 | 623.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 620.20 | 626.06 | 623.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:45:00 | 621.20 | 626.06 | 623.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 625.00 | 625.84 | 623.62 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 615.30 | 622.07 | 622.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 605.85 | 617.44 | 620.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 11:15:00 | 575.85 | 575.32 | 586.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 12:00:00 | 575.85 | 575.32 | 586.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 586.85 | 579.76 | 585.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 586.85 | 579.76 | 585.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 584.00 | 580.61 | 585.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 587.25 | 580.61 | 585.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 575.75 | 579.64 | 584.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 567.15 | 578.11 | 583.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 593.65 | 582.39 | 581.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 593.65 | 582.39 | 581.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 598.60 | 585.64 | 583.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 609.00 | 609.39 | 602.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 609.00 | 609.39 | 602.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 605.00 | 607.53 | 603.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 599.60 | 605.94 | 603.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 598.90 | 604.53 | 602.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 598.90 | 604.53 | 602.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 595.40 | 601.18 | 601.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 591.00 | 597.20 | 599.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 591.10 | 590.97 | 594.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 591.10 | 590.97 | 594.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 584.50 | 590.14 | 593.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:00:00 | 583.65 | 588.84 | 592.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 576.95 | 584.36 | 588.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 578.10 | 583.27 | 587.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:45:00 | 583.20 | 580.03 | 583.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 583.30 | 580.68 | 583.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 583.30 | 580.68 | 583.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 582.80 | 581.11 | 583.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:45:00 | 582.75 | 581.11 | 583.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 583.30 | 580.86 | 582.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 583.30 | 580.86 | 582.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 577.15 | 580.11 | 582.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:30:00 | 579.85 | 580.11 | 582.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 596.50 | 583.37 | 583.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 596.50 | 583.37 | 583.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 596.10 | 585.92 | 584.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 596.10 | 585.92 | 584.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 601.40 | 591.27 | 587.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 644.80 | 644.97 | 633.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 13:45:00 | 641.90 | 644.97 | 633.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 636.60 | 642.53 | 634.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 647.35 | 642.53 | 634.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 10:15:00 | 628.85 | 641.56 | 639.24 | SL hit (close<static) qty=1.00 sl=628.90 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 625.70 | 636.03 | 636.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 622.05 | 629.95 | 633.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 620.45 | 619.03 | 624.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 620.45 | 619.03 | 624.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 620.45 | 619.03 | 624.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 622.40 | 619.03 | 624.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 620.15 | 619.30 | 623.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 621.40 | 619.30 | 623.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 624.00 | 620.24 | 623.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 624.00 | 620.24 | 623.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 624.85 | 621.16 | 623.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 623.50 | 621.16 | 623.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 625.00 | 621.93 | 623.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:45:00 | 611.25 | 616.74 | 619.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 12:15:00 | 613.80 | 612.58 | 612.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 613.80 | 612.58 | 612.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 13:15:00 | 616.95 | 613.46 | 612.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 600.80 | 613.12 | 613.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 600.80 | 613.12 | 613.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 600.80 | 613.12 | 613.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 600.80 | 613.12 | 613.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 601.75 | 610.85 | 612.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 591.95 | 603.23 | 607.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 571.50 | 570.39 | 584.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 584.05 | 570.39 | 584.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 575.55 | 571.42 | 583.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:30:00 | 572.85 | 577.65 | 581.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 595.25 | 578.36 | 579.07 | SL hit (close>static) qty=1.00 sl=588.95 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 595.45 | 581.78 | 580.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 13:15:00 | 599.40 | 588.98 | 584.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 14:15:00 | 755.55 | 756.75 | 738.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 15:00:00 | 755.55 | 756.75 | 738.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 714.90 | 748.25 | 737.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 714.90 | 748.25 | 737.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 701.85 | 738.97 | 734.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 701.85 | 738.97 | 734.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 708.45 | 727.51 | 729.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 697.80 | 718.15 | 724.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 14:15:00 | 675.30 | 672.88 | 686.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 15:00:00 | 675.30 | 672.88 | 686.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 664.80 | 648.31 | 656.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 665.10 | 648.31 | 656.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 664.30 | 651.51 | 657.24 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 702.10 | 665.30 | 662.74 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 651.10 | 666.19 | 667.39 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 680.45 | 669.34 | 668.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 686.35 | 674.13 | 670.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 677.70 | 680.30 | 675.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 677.70 | 680.30 | 675.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 684.85 | 681.21 | 676.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 684.85 | 681.21 | 676.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 665.00 | 678.74 | 675.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:45:00 | 679.95 | 678.08 | 676.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:45:00 | 678.75 | 677.54 | 676.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 681.35 | 676.84 | 675.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-15 09:15:00 | 746.63 | 735.97 | 727.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 779.35 | 783.73 | 783.76 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 794.75 | 784.99 | 784.27 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 15:15:00 | 782.50 | 784.18 | 784.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 779.50 | 783.25 | 783.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 780.00 | 779.35 | 780.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 780.00 | 779.35 | 780.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 780.00 | 779.35 | 780.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 780.00 | 779.35 | 780.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 784.40 | 780.36 | 781.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:45:00 | 787.05 | 780.36 | 781.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 786.85 | 781.66 | 781.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:45:00 | 788.65 | 781.66 | 781.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 786.10 | 782.55 | 782.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 790.05 | 784.05 | 782.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 801.80 | 805.63 | 798.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 801.80 | 805.63 | 798.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 804.25 | 805.35 | 798.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 812.00 | 805.35 | 798.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:15:00 | 810.00 | 807.01 | 801.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 11:45:00 | 810.15 | 812.75 | 809.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 15:15:00 | 802.00 | 807.87 | 808.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 802.00 | 807.87 | 808.20 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 825.35 | 811.36 | 809.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 842.60 | 829.55 | 822.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 884.50 | 885.86 | 870.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:30:00 | 887.80 | 885.86 | 870.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 871.15 | 880.23 | 871.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 872.85 | 880.23 | 871.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 875.00 | 879.18 | 871.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 15:00:00 | 879.85 | 879.32 | 872.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 868.20 | 878.08 | 873.00 | SL hit (close<static) qty=1.00 sl=870.70 alert=retest2 |

### Cycle 76 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 850.10 | 866.96 | 868.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 847.95 | 863.15 | 866.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 820.70 | 817.40 | 826.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 820.70 | 817.40 | 826.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 829.10 | 819.67 | 826.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 829.10 | 819.67 | 826.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 821.65 | 820.06 | 825.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 823.90 | 820.06 | 825.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 826.00 | 819.14 | 822.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 826.00 | 819.14 | 822.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 825.05 | 820.32 | 822.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 827.55 | 820.32 | 822.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 819.30 | 820.48 | 822.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 815.65 | 819.60 | 821.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 816.80 | 818.98 | 820.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 815.90 | 818.98 | 820.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:00:00 | 815.95 | 818.37 | 820.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 819.90 | 817.82 | 819.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 819.90 | 817.82 | 819.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 816.00 | 817.46 | 819.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 820.00 | 817.46 | 819.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 830.80 | 820.13 | 820.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 830.80 | 820.13 | 820.16 | SL hit (close>static) qty=1.00 sl=823.50 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 826.45 | 821.39 | 820.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 848.40 | 831.26 | 826.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 861.00 | 863.91 | 854.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 861.00 | 863.91 | 854.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 859.25 | 861.65 | 856.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:45:00 | 857.65 | 861.65 | 856.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 859.20 | 861.16 | 856.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 868.05 | 861.16 | 856.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 14:15:00 | 855.30 | 863.92 | 860.94 | SL hit (close<static) qty=1.00 sl=855.50 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 851.35 | 863.63 | 865.29 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 902.30 | 866.78 | 865.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 903.95 | 874.22 | 868.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 926.00 | 927.62 | 917.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 10:00:00 | 926.00 | 927.62 | 917.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 917.40 | 923.91 | 917.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 927.50 | 921.43 | 918.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:00:00 | 923.00 | 922.21 | 919.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 923.50 | 922.51 | 919.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:00:00 | 922.60 | 923.17 | 920.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 932.90 | 934.44 | 929.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 928.85 | 934.44 | 929.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 934.10 | 934.37 | 929.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 938.00 | 934.84 | 931.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 922.45 | 929.92 | 929.76 | SL hit (close<static) qty=1.00 sl=924.55 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 922.25 | 928.39 | 929.08 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 934.80 | 930.08 | 929.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 943.70 | 933.16 | 931.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 939.80 | 940.45 | 936.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:30:00 | 938.80 | 940.45 | 936.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 936.60 | 939.68 | 936.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 936.60 | 939.68 | 936.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 936.35 | 939.01 | 936.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:30:00 | 939.20 | 938.52 | 936.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 939.00 | 938.01 | 936.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 932.80 | 936.84 | 936.64 | SL hit (close<static) qty=1.00 sl=934.20 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 930.85 | 935.64 | 936.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 927.35 | 932.67 | 934.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 924.00 | 920.34 | 926.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 924.00 | 920.34 | 926.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 931.50 | 922.57 | 926.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 931.50 | 922.57 | 926.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 935.25 | 925.11 | 927.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 935.50 | 925.11 | 927.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 933.15 | 928.52 | 928.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 933.15 | 928.52 | 928.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 932.70 | 929.36 | 928.94 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 923.80 | 927.87 | 928.31 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 935.75 | 929.75 | 929.01 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 918.00 | 928.20 | 929.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 905.35 | 924.03 | 926.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 907.00 | 907.00 | 914.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 899.65 | 907.00 | 914.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 903.65 | 896.56 | 902.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 903.65 | 896.56 | 902.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 911.50 | 899.54 | 903.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 911.50 | 899.54 | 903.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 914.90 | 902.62 | 904.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 914.90 | 902.62 | 904.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 935.90 | 909.27 | 907.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 940.85 | 915.59 | 910.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 913.05 | 924.35 | 919.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 913.05 | 924.35 | 919.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 913.05 | 924.35 | 919.15 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 911.20 | 917.41 | 917.59 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 927.85 | 918.54 | 918.02 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 914.45 | 921.37 | 921.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 908.10 | 916.14 | 918.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 915.65 | 913.44 | 916.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 12:00:00 | 915.65 | 913.44 | 916.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 916.00 | 913.95 | 916.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:15:00 | 917.20 | 913.95 | 916.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 922.00 | 915.56 | 917.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:30:00 | 921.00 | 915.56 | 917.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 925.00 | 917.45 | 917.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 926.05 | 917.45 | 917.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 922.30 | 918.42 | 918.31 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 912.55 | 917.24 | 917.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 908.00 | 915.40 | 916.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 916.65 | 909.53 | 913.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 916.65 | 909.53 | 913.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 916.65 | 909.53 | 913.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 916.65 | 909.53 | 913.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 918.00 | 911.23 | 913.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 916.15 | 911.23 | 913.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 913.00 | 911.44 | 913.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 909.45 | 911.04 | 912.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 907.75 | 907.51 | 909.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 919.50 | 911.21 | 911.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 919.50 | 911.21 | 911.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 933.00 | 919.03 | 916.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 924.70 | 926.11 | 922.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 924.70 | 926.11 | 922.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 929.40 | 926.77 | 922.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:30:00 | 930.00 | 927.75 | 923.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 930.35 | 928.79 | 924.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:30:00 | 930.00 | 929.01 | 925.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 937.50 | 928.24 | 925.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 950.60 | 957.58 | 952.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 948.55 | 957.58 | 952.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 953.25 | 956.71 | 952.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 941.00 | 949.72 | 950.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 941.00 | 949.72 | 950.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 936.05 | 945.37 | 947.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 880.05 | 874.78 | 888.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 883.70 | 874.78 | 888.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 888.65 | 879.70 | 887.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 888.65 | 879.70 | 887.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 890.25 | 881.81 | 888.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 888.90 | 881.81 | 888.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 893.70 | 884.19 | 888.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 893.70 | 884.19 | 888.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 895.00 | 886.35 | 889.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 877.30 | 886.35 | 889.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 882.80 | 880.00 | 883.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 881.65 | 880.00 | 883.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 879.65 | 879.93 | 882.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 879.75 | 879.93 | 882.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 881.95 | 878.86 | 881.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 881.95 | 878.86 | 881.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 880.50 | 879.19 | 881.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 887.40 | 879.19 | 881.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 881.60 | 879.67 | 881.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 878.90 | 879.26 | 880.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 873.50 | 874.43 | 876.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 886.05 | 876.95 | 876.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 886.05 | 876.95 | 876.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 889.60 | 879.48 | 877.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 885.00 | 889.06 | 884.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 885.00 | 889.06 | 884.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 885.00 | 889.06 | 884.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 886.00 | 889.06 | 884.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 881.85 | 887.62 | 884.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 881.85 | 887.62 | 884.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 882.00 | 886.49 | 883.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:00:00 | 885.35 | 885.87 | 884.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 895.55 | 885.15 | 884.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 920.50 | 929.00 | 929.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 920.50 | 929.00 | 929.81 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 947.75 | 930.56 | 929.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 958.20 | 936.08 | 932.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 950.75 | 952.93 | 946.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 15:00:00 | 950.75 | 952.93 | 946.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 946.90 | 951.33 | 946.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 948.95 | 951.33 | 946.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 947.85 | 950.64 | 946.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:15:00 | 952.10 | 950.47 | 946.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 944.30 | 953.30 | 951.61 | SL hit (close<static) qty=1.00 sl=945.40 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 938.75 | 948.59 | 949.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 930.70 | 943.38 | 947.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 925.45 | 919.31 | 927.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 925.45 | 919.31 | 927.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 925.45 | 919.31 | 927.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 925.45 | 919.31 | 927.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 913.45 | 918.14 | 926.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 912.40 | 916.43 | 922.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 912.55 | 915.66 | 921.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 15:15:00 | 911.00 | 904.38 | 903.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 911.00 | 904.38 | 903.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 917.95 | 907.10 | 905.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 935.00 | 936.82 | 928.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 933.20 | 936.82 | 928.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 932.35 | 935.92 | 928.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 928.00 | 935.92 | 928.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 925.00 | 933.74 | 928.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 925.00 | 933.74 | 928.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 930.55 | 933.10 | 928.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 932.10 | 930.07 | 928.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-21 13:15:00 | 1025.31 | 1018.37 | 1013.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1008.55 | 1046.89 | 1050.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 15:15:00 | 1005.10 | 1023.52 | 1037.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 10:15:00 | 1027.35 | 1023.20 | 1034.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:45:00 | 1027.90 | 1023.20 | 1034.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 996.05 | 983.44 | 997.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 997.45 | 983.44 | 997.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1004.20 | 987.59 | 998.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 1004.20 | 987.59 | 998.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1002.50 | 990.57 | 998.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 1001.00 | 1000.54 | 1001.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 1005.55 | 987.72 | 989.83 | SL hit (close>static) qty=1.00 sl=1004.65 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 995.85 | 991.79 | 991.46 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 989.00 | 994.54 | 994.86 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 996.10 | 994.25 | 994.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 1013.95 | 998.46 | 996.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 996.00 | 1003.91 | 1000.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 996.00 | 1003.91 | 1000.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 996.00 | 1003.91 | 1000.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 996.00 | 1003.91 | 1000.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 991.60 | 1001.45 | 999.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 991.60 | 1001.45 | 999.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 999.70 | 999.51 | 998.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 998.55 | 999.51 | 998.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 998.15 | 999.24 | 998.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:30:00 | 996.75 | 999.24 | 998.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 993.85 | 998.16 | 998.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 988.30 | 996.19 | 997.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 991.25 | 990.87 | 993.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:00:00 | 991.25 | 990.87 | 993.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 947.45 | 938.92 | 945.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 947.45 | 938.92 | 945.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 945.05 | 940.14 | 945.32 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 962.85 | 948.52 | 948.13 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 926.80 | 946.96 | 948.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 919.40 | 938.25 | 944.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 939.00 | 930.64 | 936.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 11:15:00 | 939.00 | 930.64 | 936.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 939.00 | 930.64 | 936.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:45:00 | 941.40 | 930.64 | 936.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 944.55 | 933.42 | 937.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:30:00 | 941.00 | 933.42 | 937.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 959.40 | 938.61 | 939.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 959.40 | 938.61 | 939.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 960.95 | 943.08 | 941.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 978.40 | 952.39 | 946.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 14:15:00 | 959.05 | 964.24 | 955.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 959.05 | 964.24 | 955.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 955.05 | 962.40 | 955.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 949.50 | 962.40 | 955.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 949.35 | 959.79 | 954.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 949.60 | 959.79 | 954.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 953.85 | 958.60 | 954.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 958.60 | 956.00 | 954.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 936.05 | 951.73 | 952.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 936.05 | 951.73 | 952.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 933.70 | 944.28 | 948.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 849.50 | 845.87 | 864.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 849.50 | 845.87 | 864.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 865.35 | 842.23 | 846.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 865.35 | 842.23 | 846.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 865.15 | 846.81 | 847.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 869.50 | 846.81 | 847.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 868.00 | 851.05 | 849.68 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 844.40 | 851.11 | 851.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 836.30 | 845.11 | 848.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 838.65 | 837.37 | 841.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 838.65 | 837.37 | 841.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 839.00 | 837.76 | 841.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:45:00 | 839.10 | 837.76 | 841.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 855.50 | 839.89 | 840.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 855.50 | 839.89 | 840.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 859.20 | 843.75 | 842.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 12:15:00 | 865.55 | 850.97 | 846.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 855.20 | 857.75 | 851.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 10:00:00 | 855.20 | 857.75 | 851.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 851.05 | 856.41 | 851.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 851.05 | 856.41 | 851.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 852.70 | 855.67 | 851.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:45:00 | 849.35 | 855.67 | 851.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 856.20 | 855.77 | 852.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:15:00 | 854.75 | 855.77 | 852.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 854.80 | 855.58 | 852.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:30:00 | 858.15 | 856.61 | 853.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 861.80 | 856.59 | 853.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 861.30 | 878.04 | 879.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 861.30 | 878.04 | 879.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 857.50 | 863.62 | 869.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 854.35 | 848.88 | 855.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 854.35 | 848.88 | 855.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 854.35 | 848.88 | 855.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 854.35 | 848.88 | 855.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 858.65 | 851.78 | 856.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 858.65 | 851.78 | 856.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 857.50 | 852.92 | 856.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 860.00 | 852.92 | 856.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 856.00 | 853.96 | 856.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 858.80 | 853.96 | 856.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 847.70 | 852.71 | 855.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:30:00 | 847.45 | 851.52 | 854.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 863.30 | 850.62 | 851.65 | SL hit (close>static) qty=1.00 sl=862.45 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 867.30 | 853.96 | 853.08 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 850.05 | 857.78 | 858.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 840.00 | 854.22 | 857.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 849.00 | 848.73 | 853.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:45:00 | 847.00 | 848.73 | 853.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 847.80 | 848.43 | 852.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 840.35 | 846.23 | 850.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 798.33 | 816.83 | 829.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 817.30 | 802.56 | 811.93 | SL hit (close>ema200) qty=0.50 sl=802.56 alert=retest2 |

### Cycle 115 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 829.10 | 817.72 | 816.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 837.80 | 821.73 | 818.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 856.90 | 859.00 | 847.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 856.90 | 859.00 | 847.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 848.00 | 855.52 | 848.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 843.00 | 855.52 | 848.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 843.00 | 853.02 | 847.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 843.00 | 853.02 | 847.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 837.05 | 849.82 | 846.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 837.05 | 849.82 | 846.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 826.20 | 841.75 | 843.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 820.60 | 831.26 | 837.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 796.75 | 787.23 | 801.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:45:00 | 797.05 | 787.23 | 801.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 782.20 | 786.22 | 799.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 794.45 | 786.22 | 799.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 784.90 | 748.10 | 757.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 784.90 | 748.10 | 757.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 802.70 | 759.02 | 761.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 802.70 | 759.02 | 761.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 800.60 | 767.33 | 765.21 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 747.75 | 767.08 | 768.32 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 781.45 | 761.21 | 760.82 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 725.00 | 754.23 | 757.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 722.85 | 747.95 | 754.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 736.50 | 735.21 | 743.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 736.10 | 735.21 | 743.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 745.10 | 737.19 | 743.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 745.10 | 737.19 | 743.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 748.70 | 739.49 | 744.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 775.00 | 739.49 | 744.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 773.55 | 746.30 | 747.06 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 790.30 | 755.10 | 750.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 794.35 | 762.95 | 754.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 780.80 | 785.10 | 777.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 780.80 | 785.10 | 777.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 780.80 | 785.10 | 777.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 780.80 | 785.10 | 777.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 773.80 | 782.84 | 776.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 773.80 | 782.84 | 776.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 768.75 | 780.02 | 776.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 768.75 | 780.02 | 776.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 772.00 | 773.79 | 773.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 758.00 | 770.63 | 772.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 772.95 | 770.66 | 772.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 772.95 | 770.66 | 772.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 772.95 | 770.66 | 772.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:45:00 | 773.15 | 770.66 | 772.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 773.00 | 771.12 | 772.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 773.85 | 771.12 | 772.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 775.65 | 772.03 | 772.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 775.65 | 772.03 | 772.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 779.20 | 773.46 | 773.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 800.80 | 779.98 | 776.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 811.35 | 811.39 | 800.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:30:00 | 809.00 | 811.39 | 800.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 798.90 | 808.42 | 804.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 800.65 | 808.42 | 804.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 803.10 | 807.36 | 804.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 805.70 | 804.28 | 803.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 15:00:00 | 804.20 | 804.22 | 803.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 786.75 | 800.85 | 802.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 786.75 | 800.85 | 802.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 12:15:00 | 781.45 | 787.42 | 792.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 786.80 | 785.72 | 790.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 787.70 | 785.85 | 789.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 781.65 | 782.95 | 786.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 775.65 | 781.46 | 783.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 771.50 | 771.68 | 776.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 736.87 | 751.76 | 760.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:15:00 | 732.92 | 741.88 | 751.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 12:15:00 | 733.00 | 732.64 | 741.26 | SL hit (close>ema200) qty=0.50 sl=732.64 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 727.25 | 715.50 | 714.59 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 690.85 | 713.27 | 714.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 687.65 | 701.16 | 708.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 706.90 | 699.11 | 704.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 706.90 | 699.11 | 704.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 706.90 | 699.11 | 704.69 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 717.85 | 707.38 | 707.25 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 702.95 | 708.10 | 708.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 699.30 | 705.55 | 706.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 711.10 | 703.25 | 705.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 711.10 | 703.25 | 705.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 711.10 | 703.25 | 705.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 711.10 | 703.25 | 705.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 706.70 | 703.94 | 705.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 13:45:00 | 701.45 | 703.63 | 705.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 666.38 | 683.04 | 691.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 679.60 | 674.92 | 681.95 | SL hit (close>ema200) qty=0.50 sl=674.92 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 694.00 | 683.92 | 683.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 702.35 | 687.61 | 685.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 681.20 | 691.81 | 688.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 681.20 | 691.81 | 688.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 681.20 | 691.81 | 688.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 678.95 | 691.81 | 688.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 678.80 | 689.21 | 687.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 682.50 | 689.21 | 687.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 676.40 | 685.13 | 686.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 674.50 | 683.01 | 685.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 680.30 | 679.21 | 682.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 680.30 | 679.21 | 682.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 680.30 | 679.21 | 682.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:00:00 | 672.95 | 678.12 | 681.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 639.30 | 658.94 | 669.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 639.75 | 638.87 | 651.71 | SL hit (close>ema200) qty=0.50 sl=638.87 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 682.30 | 656.80 | 654.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 689.05 | 663.25 | 658.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 672.15 | 675.22 | 667.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 672.15 | 675.22 | 667.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 672.15 | 675.22 | 667.08 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 644.25 | 662.48 | 664.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 642.60 | 656.16 | 661.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 676.70 | 652.65 | 657.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 676.70 | 652.65 | 657.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 676.70 | 652.65 | 657.26 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 672.90 | 660.21 | 660.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 680.05 | 668.73 | 664.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 658.05 | 668.34 | 665.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 658.05 | 668.34 | 665.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 658.05 | 668.34 | 665.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 658.05 | 668.34 | 665.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 667.95 | 668.26 | 665.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 669.70 | 668.26 | 665.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 736.67 | 706.44 | 695.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 798.70 | 810.81 | 811.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 789.00 | 806.45 | 809.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 790.65 | 785.65 | 792.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 790.65 | 785.65 | 792.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 790.65 | 785.65 | 792.51 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 799.00 | 794.26 | 794.06 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 783.00 | 793.28 | 793.89 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 798.80 | 791.46 | 791.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 11:15:00 | 806.40 | 794.45 | 792.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 798.50 | 800.31 | 796.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 798.50 | 800.31 | 796.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 798.50 | 800.31 | 796.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 798.50 | 800.31 | 796.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 807.00 | 801.65 | 797.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 823.35 | 801.65 | 797.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-31 09:15:00 | 572.00 | 2024-05-31 10:15:00 | 560.65 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-06-18 10:30:00 | 665.70 | 2024-06-21 11:15:00 | 667.95 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2024-06-19 09:15:00 | 671.35 | 2024-06-21 11:15:00 | 667.95 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-06-21 10:15:00 | 663.00 | 2024-06-21 11:15:00 | 667.95 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2024-06-27 13:15:00 | 623.55 | 2024-07-01 09:15:00 | 592.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-28 09:15:00 | 624.70 | 2024-07-01 09:15:00 | 593.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-27 13:15:00 | 623.55 | 2024-07-03 13:15:00 | 561.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-28 09:15:00 | 624.70 | 2024-07-03 13:15:00 | 562.23 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-02 13:45:00 | 665.70 | 2024-08-05 10:15:00 | 603.45 | STOP_HIT | 1.00 | -9.35% |
| BUY | retest1 | 2024-08-26 09:15:00 | 735.55 | 2024-08-26 10:15:00 | 772.33 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-26 09:15:00 | 735.55 | 2024-08-27 10:15:00 | 748.25 | STOP_HIT | 0.50 | 1.73% |
| BUY | retest2 | 2024-09-05 13:45:00 | 773.05 | 2024-09-09 09:15:00 | 729.35 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest2 | 2024-09-05 15:00:00 | 773.75 | 2024-09-09 09:15:00 | 729.35 | STOP_HIT | 1.00 | -5.74% |
| SELL | retest2 | 2024-09-10 13:15:00 | 732.35 | 2024-09-11 10:15:00 | 748.10 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-10-01 09:15:00 | 735.85 | 2024-10-03 13:15:00 | 699.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 09:15:00 | 735.85 | 2024-10-04 10:15:00 | 724.65 | STOP_HIT | 0.50 | 1.52% |
| BUY | retest2 | 2024-10-18 10:30:00 | 951.40 | 2024-10-18 14:15:00 | 1046.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-22 09:45:00 | 949.80 | 2024-10-22 12:15:00 | 933.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-25 09:30:00 | 910.55 | 2024-10-25 10:15:00 | 865.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:30:00 | 910.55 | 2024-10-28 09:15:00 | 923.15 | STOP_HIT | 0.50 | -1.38% |
| SELL | retest2 | 2024-10-28 15:00:00 | 920.00 | 2024-10-28 15:15:00 | 929.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-11-04 15:15:00 | 947.50 | 2024-11-05 10:15:00 | 926.15 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-11-05 09:45:00 | 950.45 | 2024-11-05 10:15:00 | 926.15 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-11-19 15:00:00 | 904.30 | 2024-11-22 12:15:00 | 910.80 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-11-28 09:15:00 | 926.65 | 2024-11-28 14:15:00 | 901.75 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-12-06 13:00:00 | 975.05 | 2024-12-13 11:15:00 | 986.50 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2024-12-06 15:00:00 | 973.80 | 2024-12-13 11:15:00 | 986.50 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-12-09 09:15:00 | 999.15 | 2024-12-13 11:15:00 | 986.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-12-19 14:15:00 | 954.00 | 2024-12-26 12:15:00 | 906.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 14:15:00 | 954.00 | 2024-12-26 15:15:00 | 920.30 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2025-01-21 10:15:00 | 776.65 | 2025-01-22 09:15:00 | 737.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 776.65 | 2025-01-23 09:15:00 | 746.35 | STOP_HIT | 0.50 | 3.90% |
| BUY | retest2 | 2025-02-07 09:15:00 | 692.00 | 2025-02-07 13:15:00 | 669.60 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-02-07 11:15:00 | 688.50 | 2025-02-07 13:15:00 | 669.60 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-02-19 13:00:00 | 616.85 | 2025-02-20 10:15:00 | 625.15 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-02-20 09:45:00 | 615.00 | 2025-02-20 10:15:00 | 625.15 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-03-03 11:15:00 | 567.15 | 2025-03-05 10:15:00 | 593.65 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-03-12 11:00:00 | 583.65 | 2025-03-18 10:15:00 | 596.10 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-03-13 09:15:00 | 576.95 | 2025-03-18 10:15:00 | 596.10 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-03-13 10:45:00 | 578.10 | 2025-03-18 10:15:00 | 596.10 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-03-17 09:45:00 | 583.20 | 2025-03-18 10:15:00 | 596.10 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-03-24 09:15:00 | 647.35 | 2025-03-25 10:15:00 | 628.85 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-04-01 09:45:00 | 611.25 | 2025-04-03 12:15:00 | 613.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-04-09 09:30:00 | 572.85 | 2025-04-11 09:15:00 | 595.25 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-05-09 11:45:00 | 679.95 | 2025-05-15 09:15:00 | 746.63 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2025-05-09 12:45:00 | 678.75 | 2025-05-15 11:15:00 | 747.95 | TARGET_HIT | 1.00 | 10.19% |
| BUY | retest2 | 2025-05-09 15:15:00 | 681.35 | 2025-05-15 12:15:00 | 749.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-30 11:15:00 | 812.00 | 2025-06-03 15:15:00 | 802.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-05-30 14:15:00 | 810.00 | 2025-06-03 15:15:00 | 802.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-03 11:45:00 | 810.15 | 2025-06-03 15:15:00 | 802.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-10 15:00:00 | 879.85 | 2025-06-11 09:15:00 | 868.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-18 15:15:00 | 815.65 | 2025-06-20 09:15:00 | 830.80 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-06-19 09:30:00 | 816.80 | 2025-06-20 09:15:00 | 830.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-06-19 10:15:00 | 815.90 | 2025-06-20 09:15:00 | 830.80 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-06-19 11:00:00 | 815.95 | 2025-06-20 09:15:00 | 830.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-06-27 09:15:00 | 868.05 | 2025-06-27 14:15:00 | 855.30 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-06-30 09:15:00 | 868.90 | 2025-07-02 09:15:00 | 854.85 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-09 09:15:00 | 927.50 | 2025-07-14 11:15:00 | 922.45 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-09 12:00:00 | 923.00 | 2025-07-14 12:15:00 | 922.25 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-07-09 13:30:00 | 923.50 | 2025-07-14 12:15:00 | 922.25 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-07-10 10:00:00 | 922.60 | 2025-07-14 12:15:00 | 922.25 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-07-11 14:45:00 | 938.00 | 2025-07-14 12:15:00 | 922.25 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-16 14:30:00 | 939.20 | 2025-07-17 12:15:00 | 932.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-17 09:15:00 | 939.00 | 2025-07-17 12:15:00 | 932.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-08 14:45:00 | 909.45 | 2025-08-11 14:15:00 | 919.50 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-11 13:15:00 | 907.75 | 2025-08-11 14:15:00 | 919.50 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-08-14 11:30:00 | 930.00 | 2025-08-21 13:15:00 | 941.00 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-08-14 13:30:00 | 930.35 | 2025-08-21 13:15:00 | 941.00 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-08-14 14:30:00 | 930.00 | 2025-08-21 13:15:00 | 941.00 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-08-18 09:15:00 | 937.50 | 2025-08-21 13:15:00 | 941.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-09-04 10:30:00 | 878.90 | 2025-09-08 10:15:00 | 886.05 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-05 10:45:00 | 873.50 | 2025-09-08 10:15:00 | 886.05 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-09 14:00:00 | 885.35 | 2025-09-18 12:15:00 | 920.50 | STOP_HIT | 1.00 | 3.97% |
| BUY | retest2 | 2025-09-10 09:15:00 | 895.55 | 2025-09-18 12:15:00 | 920.50 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest2 | 2025-09-23 12:15:00 | 952.10 | 2025-09-24 14:15:00 | 944.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-30 09:15:00 | 912.40 | 2025-10-03 15:15:00 | 911.00 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-09-30 10:00:00 | 912.55 | 2025-10-03 15:15:00 | 911.00 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-10-09 09:30:00 | 932.10 | 2025-10-21 13:15:00 | 1025.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-04 11:15:00 | 1001.00 | 2025-11-07 12:15:00 | 1005.55 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-02 13:45:00 | 958.60 | 2025-12-03 09:15:00 | 936.05 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-12-19 14:30:00 | 858.15 | 2025-12-26 14:15:00 | 861.30 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-12-22 09:15:00 | 861.80 | 2025-12-26 14:15:00 | 861.30 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2026-01-01 10:30:00 | 847.45 | 2026-01-02 10:15:00 | 863.30 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-01-08 10:45:00 | 840.35 | 2026-01-09 14:15:00 | 798.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 840.35 | 2026-01-13 09:15:00 | 817.30 | STOP_HIT | 0.50 | 2.74% |
| BUY | retest2 | 2026-02-12 13:15:00 | 805.70 | 2026-02-13 09:15:00 | 786.75 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-02-12 15:00:00 | 804.20 | 2026-02-13 09:15:00 | 786.75 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-02-19 09:15:00 | 775.65 | 2026-02-24 14:15:00 | 736.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:00:00 | 771.50 | 2026-02-25 12:15:00 | 732.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 775.65 | 2026-02-26 12:15:00 | 733.00 | STOP_HIT | 0.50 | 5.50% |
| SELL | retest2 | 2026-02-20 11:00:00 | 771.50 | 2026-02-26 12:15:00 | 733.00 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2026-03-12 13:45:00 | 701.45 | 2026-03-16 09:15:00 | 666.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 13:45:00 | 701.45 | 2026-03-17 09:15:00 | 679.60 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2026-03-20 12:00:00 | 672.95 | 2026-03-23 10:15:00 | 639.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:00:00 | 672.95 | 2026-03-24 10:15:00 | 639.75 | STOP_HIT | 0.50 | 4.93% |
| BUY | retest2 | 2026-04-02 11:15:00 | 669.70 | 2026-04-08 09:15:00 | 736.67 | TARGET_HIT | 1.00 | 10.00% |
