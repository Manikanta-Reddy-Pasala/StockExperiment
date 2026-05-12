# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1342.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 155 |
| ALERT1 | 106 |
| ALERT2 | 102 |
| ALERT2_SKIP | 46 |
| ALERT3 | 268 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 117 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 112 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 139 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 55 / 84
- **Target hits / Stop hits / Partials:** 8 / 112 / 19
- **Avg / median % per leg:** 0.42% / -1.12%
- **Sum % (uncompounded):** 59.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 16 | 26.7% | 5 | 55 | 0 | -0.42% | -25.1% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.65% | -3.3% |
| BUY @ 3rd Alert (retest2) | 58 | 16 | 27.6% | 5 | 53 | 0 | -0.38% | -21.8% |
| SELL (all) | 79 | 39 | 49.4% | 3 | 57 | 19 | 1.07% | 84.2% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | 0.03% | 0.1% |
| SELL @ 3rd Alert (retest2) | 74 | 38 | 51.4% | 3 | 53 | 18 | 1.14% | 84.0% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.45% | -3.2% |
| retest2 (combined) | 132 | 54 | 40.9% | 8 | 106 | 18 | 0.47% | 62.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 550.60 | 532.99 | 532.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 562.20 | 538.83 | 535.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 556.55 | 557.12 | 552.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 556.55 | 557.12 | 552.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 556.10 | 557.06 | 553.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:45:00 | 556.10 | 557.06 | 553.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 556.00 | 556.85 | 553.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 555.05 | 556.85 | 553.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 549.60 | 555.36 | 553.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 548.65 | 555.36 | 553.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 547.40 | 553.77 | 553.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 543.70 | 553.77 | 553.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 552.65 | 553.46 | 553.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 552.65 | 553.46 | 553.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 551.55 | 553.08 | 552.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 551.55 | 553.08 | 552.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 552.20 | 552.90 | 552.91 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 12:15:00 | 555.65 | 552.87 | 552.81 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 551.55 | 552.72 | 552.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 15:15:00 | 548.50 | 551.42 | 552.19 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 566.05 | 554.34 | 553.45 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 547.20 | 554.90 | 555.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 544.65 | 551.54 | 553.99 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 574.50 | 551.45 | 551.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 580.00 | 569.13 | 562.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 14:15:00 | 564.85 | 570.03 | 564.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 564.85 | 570.03 | 564.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 564.85 | 570.03 | 564.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 564.85 | 570.03 | 564.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 565.00 | 569.02 | 564.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 570.30 | 569.02 | 564.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 11:30:00 | 565.40 | 567.71 | 565.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 12:15:00 | 565.25 | 567.71 | 565.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 550.45 | 563.70 | 564.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 550.45 | 563.70 | 564.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 531.50 | 557.26 | 561.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 539.90 | 539.61 | 548.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:15:00 | 542.90 | 539.61 | 548.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 546.95 | 542.93 | 548.10 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 573.05 | 553.30 | 551.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 576.10 | 565.60 | 558.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 10:15:00 | 605.00 | 605.70 | 597.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:45:00 | 606.00 | 605.70 | 597.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 600.00 | 603.72 | 599.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 600.00 | 603.72 | 599.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 600.00 | 602.98 | 599.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:15:00 | 596.25 | 602.98 | 599.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 600.95 | 602.57 | 599.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:30:00 | 596.90 | 602.57 | 599.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 597.55 | 601.57 | 599.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 597.55 | 601.57 | 599.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 598.30 | 600.91 | 599.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:30:00 | 599.65 | 600.92 | 599.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:00:00 | 600.95 | 600.92 | 599.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:30:00 | 600.00 | 601.54 | 599.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 14:15:00 | 593.25 | 600.99 | 600.47 | SL hit (close<static) qty=1.00 sl=594.10 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 15:15:00 | 595.00 | 599.79 | 599.98 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 601.95 | 600.12 | 600.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 11:15:00 | 608.90 | 601.88 | 600.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 699.20 | 699.56 | 677.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 699.20 | 699.56 | 677.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 686.75 | 694.49 | 680.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 683.45 | 694.49 | 680.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 684.35 | 692.46 | 680.81 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 666.60 | 676.93 | 677.81 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 687.10 | 678.96 | 678.66 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 671.00 | 678.63 | 678.86 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 690.65 | 681.03 | 679.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 712.00 | 690.57 | 685.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 09:15:00 | 686.20 | 690.73 | 686.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 686.20 | 690.73 | 686.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 686.20 | 690.73 | 686.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 686.20 | 690.73 | 686.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 681.70 | 688.92 | 685.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 680.85 | 688.92 | 685.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 679.30 | 687.00 | 685.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:30:00 | 680.05 | 687.00 | 685.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 666.50 | 682.90 | 683.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 656.95 | 677.71 | 681.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 13:15:00 | 676.05 | 665.35 | 671.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 13:15:00 | 676.05 | 665.35 | 671.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 676.05 | 665.35 | 671.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:00:00 | 676.05 | 665.35 | 671.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 676.05 | 667.49 | 671.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:30:00 | 680.00 | 667.49 | 671.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 692.00 | 674.71 | 674.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 702.20 | 687.48 | 682.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 13:15:00 | 752.50 | 758.24 | 745.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 13:45:00 | 749.70 | 758.24 | 745.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 766.10 | 759.48 | 749.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 14:30:00 | 772.20 | 764.68 | 755.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 15:00:00 | 772.00 | 764.68 | 755.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 788.00 | 765.36 | 756.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 746.55 | 770.90 | 766.32 | SL hit (close<static) qty=1.00 sl=747.55 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 744.75 | 760.56 | 762.10 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 764.00 | 762.23 | 762.17 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 13:15:00 | 761.50 | 762.08 | 762.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 14:15:00 | 753.20 | 760.31 | 761.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 769.20 | 761.56 | 761.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 769.20 | 761.56 | 761.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 769.20 | 761.56 | 761.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 777.55 | 761.56 | 761.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 767.70 | 762.79 | 762.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 13:15:00 | 785.80 | 768.29 | 764.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 777.35 | 782.46 | 777.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 12:15:00 | 777.35 | 782.46 | 777.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 777.35 | 782.46 | 777.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:00:00 | 777.35 | 782.46 | 777.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 772.80 | 780.53 | 777.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 772.80 | 780.53 | 777.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 769.65 | 778.35 | 776.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 767.65 | 778.35 | 776.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 768.00 | 776.28 | 775.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 758.20 | 776.28 | 775.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 757.00 | 772.43 | 774.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 746.15 | 767.17 | 771.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 765.00 | 765.00 | 769.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 12:15:00 | 765.00 | 765.00 | 769.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 765.00 | 765.00 | 769.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 765.00 | 765.00 | 769.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 769.25 | 765.85 | 769.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 769.25 | 765.85 | 769.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 755.75 | 763.83 | 768.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 15:15:00 | 755.05 | 763.83 | 768.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:00:00 | 745.05 | 758.67 | 765.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 15:15:00 | 745.00 | 756.83 | 761.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 10:15:00 | 786.00 | 763.08 | 763.13 | SL hit (close>static) qty=1.00 sl=769.10 alert=retest2 |

### Cycle 23 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 778.30 | 766.12 | 764.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 15:15:00 | 790.00 | 776.05 | 772.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 925.05 | 949.80 | 928.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 14:15:00 | 925.05 | 949.80 | 928.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 925.05 | 949.80 | 928.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 925.05 | 949.80 | 928.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 942.00 | 948.24 | 930.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 959.10 | 952.47 | 933.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 956.55 | 942.75 | 936.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:30:00 | 943.25 | 942.43 | 937.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 12:15:00 | 954.75 | 942.32 | 937.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 952.45 | 953.96 | 946.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:30:00 | 936.90 | 953.96 | 946.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 927.60 | 948.69 | 944.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 927.60 | 948.69 | 944.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 921.70 | 943.29 | 942.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 921.70 | 943.29 | 942.69 | SL hit (close<static) qty=1.00 sl=923.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 925.70 | 939.77 | 941.14 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 991.90 | 947.62 | 943.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 1009.00 | 986.44 | 971.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 1001.15 | 1003.78 | 987.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 1001.15 | 1003.78 | 987.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 984.95 | 998.39 | 987.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 984.95 | 998.39 | 987.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 983.00 | 995.32 | 987.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1002.80 | 995.32 | 987.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 13:45:00 | 990.90 | 992.39 | 988.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 14:45:00 | 992.85 | 992.67 | 989.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:00:00 | 993.25 | 992.65 | 989.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1018.60 | 997.84 | 992.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-13 12:15:00 | 976.00 | 992.77 | 995.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 976.00 | 992.77 | 995.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 971.50 | 988.51 | 992.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 965.55 | 956.21 | 963.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 965.55 | 956.21 | 963.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 965.55 | 956.21 | 963.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 965.55 | 956.21 | 963.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 949.70 | 954.91 | 962.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 11:15:00 | 945.75 | 953.73 | 958.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 994.30 | 965.47 | 962.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 994.30 | 965.47 | 962.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 1013.25 | 975.02 | 966.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1046.10 | 1047.93 | 1026.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:30:00 | 1046.70 | 1047.93 | 1026.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 1074.90 | 1086.13 | 1075.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:30:00 | 1070.95 | 1086.13 | 1075.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1073.00 | 1083.50 | 1075.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1085.00 | 1083.50 | 1075.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 1070.00 | 1078.09 | 1074.50 | SL hit (close<static) qty=1.00 sl=1072.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 1067.20 | 1072.26 | 1072.39 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 09:15:00 | 1089.95 | 1075.27 | 1073.71 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 1058.60 | 1070.77 | 1071.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 1049.10 | 1064.41 | 1068.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 1061.10 | 1055.45 | 1062.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 1061.10 | 1055.45 | 1062.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1061.10 | 1055.45 | 1062.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 1068.10 | 1055.45 | 1062.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1055.00 | 1055.36 | 1061.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:15:00 | 1051.90 | 1055.36 | 1061.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 1044.70 | 1054.25 | 1059.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 09:15:00 | 1085.00 | 1058.74 | 1060.63 | SL hit (close>static) qty=1.00 sl=1064.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 14:15:00 | 1069.95 | 1063.15 | 1062.30 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 1052.35 | 1060.27 | 1061.22 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 1080.00 | 1060.38 | 1060.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 10:15:00 | 1096.95 | 1067.69 | 1063.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 09:15:00 | 1085.75 | 1086.28 | 1076.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 10:00:00 | 1085.75 | 1086.28 | 1076.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1072.80 | 1083.58 | 1076.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 1072.80 | 1083.58 | 1076.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 1063.90 | 1079.64 | 1075.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:00:00 | 1063.90 | 1079.64 | 1075.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1070.55 | 1074.88 | 1073.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:30:00 | 1071.75 | 1074.88 | 1073.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1050.90 | 1069.51 | 1071.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1021.30 | 1049.57 | 1059.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 13:15:00 | 1054.90 | 1048.83 | 1055.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 13:15:00 | 1054.90 | 1048.83 | 1055.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1054.90 | 1048.83 | 1055.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 1058.45 | 1048.83 | 1055.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1057.30 | 1050.53 | 1055.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 1053.75 | 1050.53 | 1055.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1069.00 | 1054.22 | 1057.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1077.00 | 1054.22 | 1057.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1071.00 | 1057.58 | 1058.35 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 1068.70 | 1059.80 | 1059.30 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 10:15:00 | 1048.30 | 1058.65 | 1059.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 12:15:00 | 1039.15 | 1052.77 | 1056.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 979.00 | 967.90 | 979.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 979.00 | 967.90 | 979.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 979.00 | 967.90 | 979.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:45:00 | 985.60 | 967.90 | 979.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 983.05 | 970.93 | 979.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:30:00 | 983.45 | 970.93 | 979.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 985.80 | 973.91 | 980.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:45:00 | 987.55 | 973.91 | 980.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 1016.10 | 989.05 | 986.25 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 973.40 | 984.70 | 985.42 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1028.95 | 987.29 | 983.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 1029.85 | 1011.52 | 997.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 14:15:00 | 1061.50 | 1064.81 | 1046.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 14:45:00 | 1060.00 | 1064.81 | 1046.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1069.00 | 1065.60 | 1050.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 1047.00 | 1065.60 | 1050.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 1057.20 | 1069.00 | 1061.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:00:00 | 1057.20 | 1069.00 | 1061.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 1058.40 | 1066.88 | 1061.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:00:00 | 1058.40 | 1066.88 | 1061.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1055.20 | 1064.54 | 1061.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 1059.05 | 1063.45 | 1060.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 1039.35 | 1093.13 | 1100.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 1039.35 | 1093.13 | 1100.44 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 1080.00 | 1074.13 | 1073.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1083.80 | 1077.77 | 1075.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 1064.80 | 1075.18 | 1074.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 10:15:00 | 1064.80 | 1075.18 | 1074.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 1064.80 | 1075.18 | 1074.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 1064.80 | 1075.18 | 1074.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 1046.15 | 1069.37 | 1071.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 11:15:00 | 1036.20 | 1050.29 | 1058.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1050.00 | 1044.16 | 1051.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1050.00 | 1044.16 | 1051.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1050.00 | 1044.16 | 1051.37 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 1065.90 | 1050.03 | 1049.89 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 1037.70 | 1047.80 | 1048.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 1031.90 | 1042.17 | 1046.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 991.90 | 980.32 | 994.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 991.90 | 980.32 | 994.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 991.90 | 980.32 | 994.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 991.90 | 980.32 | 994.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1023.25 | 988.90 | 996.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 1023.25 | 988.90 | 996.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1042.75 | 999.67 | 1001.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 1046.35 | 999.67 | 1001.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 1093.85 | 1018.51 | 1009.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 14:15:00 | 1124.50 | 1051.92 | 1027.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 1081.10 | 1102.31 | 1075.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 1081.10 | 1102.31 | 1075.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1081.10 | 1102.31 | 1075.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 1081.10 | 1102.31 | 1075.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 1073.80 | 1096.61 | 1075.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 1073.25 | 1096.61 | 1075.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 1071.80 | 1091.65 | 1075.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:30:00 | 1074.15 | 1091.65 | 1075.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 1082.50 | 1089.82 | 1075.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 12:30:00 | 1062.35 | 1089.82 | 1075.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 1111.15 | 1093.07 | 1079.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 15:15:00 | 1155.00 | 1093.07 | 1079.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-29 15:15:00 | 1270.50 | 1186.91 | 1148.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1313.10 | 1339.70 | 1341.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 1299.15 | 1326.40 | 1334.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1319.80 | 1306.88 | 1320.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1319.80 | 1306.88 | 1320.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1319.80 | 1306.88 | 1320.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 1314.25 | 1306.88 | 1320.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1315.85 | 1308.67 | 1319.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:45:00 | 1299.85 | 1313.00 | 1319.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:15:00 | 1234.86 | 1287.21 | 1304.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 1285.20 | 1261.51 | 1280.78 | SL hit (close>ema200) qty=0.50 sl=1261.51 alert=retest2 |

### Cycle 47 — BUY (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 10:15:00 | 1315.95 | 1292.17 | 1289.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1337.45 | 1299.83 | 1294.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 13:15:00 | 1293.00 | 1309.55 | 1302.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 13:15:00 | 1293.00 | 1309.55 | 1302.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 1293.00 | 1309.55 | 1302.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 1293.00 | 1309.55 | 1302.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1268.30 | 1301.30 | 1299.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 1268.30 | 1301.30 | 1299.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 1277.00 | 1296.44 | 1297.07 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 1306.50 | 1289.65 | 1287.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 1315.00 | 1301.44 | 1295.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 1306.25 | 1309.91 | 1302.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 14:15:00 | 1306.25 | 1309.91 | 1302.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 1306.25 | 1309.91 | 1302.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 1306.25 | 1309.91 | 1302.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1311.00 | 1310.13 | 1303.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 1313.70 | 1310.13 | 1303.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:30:00 | 1315.50 | 1311.13 | 1304.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 13:15:00 | 1355.00 | 1365.16 | 1365.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 13:15:00 | 1355.00 | 1365.16 | 1365.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 1348.45 | 1361.82 | 1363.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 09:15:00 | 1365.55 | 1360.24 | 1362.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 1365.55 | 1360.24 | 1362.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1365.55 | 1360.24 | 1362.63 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 1375.00 | 1366.13 | 1364.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 14:15:00 | 1427.00 | 1382.77 | 1374.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 13:15:00 | 1394.25 | 1395.97 | 1386.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 13:30:00 | 1389.50 | 1395.97 | 1386.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1379.00 | 1392.58 | 1385.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 1379.00 | 1392.58 | 1385.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 1383.00 | 1390.66 | 1385.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 1368.35 | 1390.66 | 1385.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1397.45 | 1388.54 | 1385.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 11:30:00 | 1406.65 | 1391.40 | 1386.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 15:15:00 | 1380.00 | 1383.50 | 1383.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 1380.00 | 1383.50 | 1383.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 1369.95 | 1380.79 | 1382.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 1332.55 | 1319.95 | 1336.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 1332.55 | 1319.95 | 1336.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1332.55 | 1319.95 | 1336.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:30:00 | 1336.25 | 1319.95 | 1336.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1321.55 | 1304.11 | 1316.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 1321.55 | 1304.11 | 1316.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1315.65 | 1306.42 | 1316.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:15:00 | 1305.95 | 1307.30 | 1315.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 1240.65 | 1268.34 | 1288.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 1238.50 | 1230.11 | 1259.15 | SL hit (close>ema200) qty=0.50 sl=1230.11 alert=retest2 |

### Cycle 53 — BUY (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 15:15:00 | 1204.00 | 1187.37 | 1186.21 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 10:15:00 | 1158.95 | 1185.95 | 1188.04 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 1195.40 | 1184.82 | 1184.06 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 1175.70 | 1182.99 | 1183.30 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1189.85 | 1184.36 | 1183.90 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 1171.95 | 1182.62 | 1183.23 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 1203.95 | 1187.06 | 1184.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 1216.25 | 1199.61 | 1192.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 1208.10 | 1210.56 | 1202.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 11:00:00 | 1208.10 | 1210.56 | 1202.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1213.00 | 1211.05 | 1203.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:00:00 | 1223.05 | 1214.38 | 1206.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:15:00 | 1223.00 | 1215.02 | 1207.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 11:00:00 | 1219.25 | 1217.31 | 1210.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 14:15:00 | 1201.80 | 1211.59 | 1209.84 | SL hit (close<static) qty=1.00 sl=1203.05 alert=retest2 |

### Cycle 60 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1175.90 | 1204.20 | 1206.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1155.80 | 1194.52 | 1202.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1184.45 | 1174.51 | 1186.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:00:00 | 1184.45 | 1174.51 | 1186.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1197.10 | 1179.02 | 1187.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1197.10 | 1179.02 | 1187.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1178.60 | 1178.94 | 1186.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 1187.65 | 1178.94 | 1186.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1187.85 | 1179.29 | 1185.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 1187.85 | 1179.29 | 1185.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1186.65 | 1180.76 | 1185.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1168.65 | 1183.59 | 1186.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 1208.25 | 1174.35 | 1177.57 | SL hit (close>static) qty=1.00 sl=1197.15 alert=retest2 |

### Cycle 61 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 10:15:00 | 1209.65 | 1181.41 | 1180.49 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 1142.50 | 1175.89 | 1179.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1108.70 | 1143.75 | 1159.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 1104.00 | 1097.75 | 1121.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 1104.00 | 1097.75 | 1121.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1108.70 | 1099.94 | 1120.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 1108.70 | 1099.94 | 1120.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 1126.00 | 1104.84 | 1118.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 1126.00 | 1104.84 | 1118.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1130.15 | 1109.90 | 1119.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 1129.15 | 1109.90 | 1119.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 1141.00 | 1122.59 | 1123.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 1133.00 | 1122.59 | 1123.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 1139.05 | 1125.88 | 1125.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 12:15:00 | 1145.15 | 1129.74 | 1127.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 12:15:00 | 1169.80 | 1177.87 | 1163.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 13:00:00 | 1169.80 | 1177.87 | 1163.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 1172.70 | 1187.34 | 1179.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:00:00 | 1172.70 | 1187.34 | 1179.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 1176.10 | 1185.09 | 1179.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 1189.15 | 1185.09 | 1179.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1157.55 | 1177.82 | 1176.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1157.55 | 1177.82 | 1176.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1155.80 | 1172.31 | 1174.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1145.00 | 1166.21 | 1170.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1157.85 | 1151.81 | 1160.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1157.85 | 1151.81 | 1160.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 1152.00 | 1151.85 | 1159.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 1162.80 | 1151.85 | 1159.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1200.65 | 1161.61 | 1163.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1200.65 | 1161.61 | 1163.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 1204.30 | 1170.15 | 1167.24 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1139.10 | 1167.43 | 1170.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 1132.40 | 1160.42 | 1167.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 1124.00 | 1118.18 | 1136.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 1124.00 | 1118.18 | 1136.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1136.60 | 1121.86 | 1136.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:45:00 | 1135.20 | 1121.86 | 1136.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 1120.05 | 1121.50 | 1135.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 1139.35 | 1121.50 | 1135.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1179.05 | 1133.55 | 1138.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1179.05 | 1133.55 | 1138.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 1189.55 | 1144.75 | 1142.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 1193.15 | 1154.43 | 1147.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 1165.65 | 1213.11 | 1186.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 10:15:00 | 1165.65 | 1213.11 | 1186.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1165.65 | 1213.11 | 1186.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:00:00 | 1165.65 | 1213.11 | 1186.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 1137.55 | 1198.00 | 1181.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:45:00 | 1153.00 | 1198.00 | 1181.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1120.25 | 1171.74 | 1172.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 15:15:00 | 1113.00 | 1123.57 | 1131.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1144.80 | 1127.82 | 1133.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 1144.80 | 1127.82 | 1133.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1144.80 | 1127.82 | 1133.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 1144.80 | 1127.82 | 1133.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1143.00 | 1130.85 | 1133.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 1139.40 | 1130.85 | 1133.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 1159.00 | 1136.62 | 1135.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 1159.00 | 1136.62 | 1135.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 14:15:00 | 1166.00 | 1149.17 | 1142.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 1141.35 | 1165.29 | 1158.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 1141.35 | 1165.29 | 1158.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1141.35 | 1165.29 | 1158.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:45:00 | 1142.45 | 1165.29 | 1158.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1140.00 | 1160.23 | 1156.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1140.00 | 1160.23 | 1156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1126.60 | 1153.50 | 1153.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 1122.55 | 1147.31 | 1151.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1100.50 | 1069.00 | 1086.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 1100.50 | 1069.00 | 1086.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1100.50 | 1069.00 | 1086.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 1098.70 | 1069.00 | 1086.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1084.55 | 1072.11 | 1085.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:15:00 | 1080.75 | 1072.11 | 1085.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 1081.30 | 1076.31 | 1085.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 1026.71 | 1062.37 | 1075.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 1027.23 | 1062.37 | 1075.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-17 09:15:00 | 972.68 | 1009.88 | 1038.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 71 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 1033.20 | 998.27 | 996.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 1038.85 | 1018.88 | 1007.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 1065.10 | 1066.61 | 1050.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 1057.30 | 1066.61 | 1050.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1056.75 | 1064.64 | 1051.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 1069.85 | 1063.33 | 1051.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 1084.65 | 1063.05 | 1056.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 1044.60 | 1055.47 | 1056.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 1044.60 | 1055.47 | 1056.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 1036.65 | 1051.71 | 1054.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 15:15:00 | 1036.75 | 1035.57 | 1044.03 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:15:00 | 972.35 | 1035.57 | 1044.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 923.73 | 957.93 | 990.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 981.00 | 940.11 | 961.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 981.00 | 940.11 | 961.13 | SL hit (close>ema200) qty=0.50 sl=940.11 alert=retest1 |

### Cycle 73 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1000.00 | 975.79 | 972.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 1024.80 | 991.87 | 980.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 1096.80 | 1105.64 | 1086.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:00:00 | 1096.80 | 1105.64 | 1086.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1099.00 | 1103.44 | 1089.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1078.90 | 1103.44 | 1089.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1076.00 | 1097.95 | 1087.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 1072.05 | 1097.95 | 1087.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1108.55 | 1100.07 | 1089.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 1085.80 | 1100.07 | 1089.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1114.45 | 1133.55 | 1126.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 1114.45 | 1133.55 | 1126.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1123.00 | 1131.44 | 1126.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:30:00 | 1128.10 | 1130.32 | 1126.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 10:30:00 | 1131.10 | 1130.00 | 1126.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 1115.75 | 1123.30 | 1123.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 13:15:00 | 1115.75 | 1123.30 | 1123.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-18 09:15:00 | 1104.20 | 1115.17 | 1119.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 10:15:00 | 1123.70 | 1116.88 | 1120.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 10:15:00 | 1123.70 | 1116.88 | 1120.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1123.70 | 1116.88 | 1120.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 1123.70 | 1116.88 | 1120.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 1119.40 | 1117.38 | 1119.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:15:00 | 1117.00 | 1117.38 | 1119.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 13:15:00 | 1112.35 | 1118.93 | 1120.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 09:30:00 | 1110.50 | 1101.31 | 1107.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 11:00:00 | 1116.90 | 1104.43 | 1108.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 1100.90 | 1103.72 | 1107.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 1118.15 | 1111.26 | 1110.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 09:15:00 | 1118.15 | 1111.26 | 1110.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 13:15:00 | 1141.70 | 1118.51 | 1114.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 13:15:00 | 1135.05 | 1139.15 | 1129.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 13:45:00 | 1139.40 | 1139.15 | 1129.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 1127.05 | 1136.73 | 1129.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:30:00 | 1117.55 | 1136.73 | 1129.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 1122.95 | 1133.97 | 1128.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 1131.95 | 1133.97 | 1128.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 13:15:00 | 1112.00 | 1142.07 | 1138.19 | SL hit (close<static) qty=1.00 sl=1121.10 alert=retest2 |

### Cycle 76 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 1124.60 | 1134.92 | 1135.39 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 1142.10 | 1136.60 | 1135.93 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 13:15:00 | 1130.45 | 1135.37 | 1135.43 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 1138.00 | 1135.89 | 1135.66 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 09:15:00 | 1129.70 | 1135.31 | 1135.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 10:15:00 | 1124.00 | 1133.05 | 1134.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1112.65 | 1109.14 | 1116.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 10:00:00 | 1112.65 | 1109.14 | 1116.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1129.75 | 1113.27 | 1117.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:45:00 | 1132.90 | 1113.27 | 1117.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1137.05 | 1118.02 | 1119.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 1137.05 | 1118.02 | 1119.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 1152.85 | 1124.99 | 1122.42 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 1134.00 | 1140.69 | 1140.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1034.00 | 1119.35 | 1131.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 1094.00 | 1084.27 | 1105.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 1094.00 | 1084.27 | 1105.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1105.85 | 1090.93 | 1104.98 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 1141.80 | 1116.33 | 1113.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 1150.00 | 1123.07 | 1116.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 1256.40 | 1260.71 | 1233.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 14:30:00 | 1248.50 | 1260.71 | 1233.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 1276.10 | 1299.90 | 1290.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 1276.10 | 1299.90 | 1290.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 1272.50 | 1294.42 | 1288.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 1295.40 | 1294.42 | 1288.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 1284.90 | 1290.92 | 1287.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:00:00 | 1284.20 | 1289.57 | 1287.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 15:00:00 | 1284.20 | 1287.00 | 1286.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1295.80 | 1288.60 | 1287.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:30:00 | 1300.40 | 1292.40 | 1289.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:15:00 | 1301.80 | 1295.15 | 1294.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 09:15:00 | 1301.90 | 1301.90 | 1298.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:45:00 | 1314.00 | 1312.27 | 1307.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1308.10 | 1311.43 | 1307.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 1302.60 | 1311.43 | 1307.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1305.50 | 1310.25 | 1307.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 1305.50 | 1310.25 | 1307.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 1297.00 | 1307.60 | 1306.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:45:00 | 1300.10 | 1307.60 | 1306.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 1303.50 | 1306.23 | 1306.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:15:00 | 1302.00 | 1306.23 | 1306.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-29 15:15:00 | 1302.00 | 1305.39 | 1305.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 15:15:00 | 1302.00 | 1305.39 | 1305.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 1283.10 | 1298.87 | 1302.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1295.00 | 1276.25 | 1283.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1295.00 | 1276.25 | 1283.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1295.00 | 1276.25 | 1283.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1295.00 | 1276.25 | 1283.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1293.50 | 1279.70 | 1284.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:15:00 | 1278.80 | 1279.70 | 1284.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1269.90 | 1278.96 | 1281.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1214.86 | 1255.46 | 1267.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1206.40 | 1255.46 | 1267.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 1259.20 | 1255.97 | 1265.68 | SL hit (close>ema200) qty=0.50 sl=1255.97 alert=retest2 |

### Cycle 85 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1285.80 | 1271.39 | 1269.45 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 1252.90 | 1267.41 | 1268.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1224.00 | 1255.36 | 1262.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 1254.00 | 1248.64 | 1256.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:00:00 | 1254.00 | 1248.64 | 1256.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 1254.50 | 1249.81 | 1256.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:15:00 | 1251.10 | 1249.81 | 1256.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 1251.10 | 1250.07 | 1255.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 1286.00 | 1250.07 | 1255.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1295.00 | 1259.06 | 1259.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 1282.70 | 1259.06 | 1259.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1320.10 | 1271.26 | 1264.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 1335.10 | 1311.14 | 1298.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 11:15:00 | 1341.70 | 1343.02 | 1330.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 12:00:00 | 1341.70 | 1343.02 | 1330.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1379.00 | 1377.96 | 1366.11 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 1353.40 | 1365.42 | 1366.57 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 1382.00 | 1368.74 | 1367.98 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 1336.30 | 1362.25 | 1365.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 10:15:00 | 1309.90 | 1348.54 | 1358.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 14:15:00 | 1328.50 | 1325.55 | 1342.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 1328.50 | 1325.55 | 1342.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1353.70 | 1330.61 | 1341.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 1353.00 | 1330.61 | 1341.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1339.20 | 1332.33 | 1341.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:30:00 | 1348.60 | 1332.33 | 1341.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1360.40 | 1337.94 | 1343.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 1360.40 | 1337.94 | 1343.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1359.00 | 1342.16 | 1344.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 1363.60 | 1342.16 | 1344.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1364.80 | 1349.38 | 1347.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1381.50 | 1358.91 | 1354.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 11:15:00 | 1484.00 | 1485.94 | 1452.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 11:30:00 | 1480.00 | 1485.94 | 1452.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1491.00 | 1485.47 | 1464.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 1534.50 | 1482.72 | 1472.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 1519.70 | 1537.70 | 1539.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 1519.70 | 1537.70 | 1539.17 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 1560.90 | 1539.25 | 1538.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 10:15:00 | 1566.50 | 1544.70 | 1541.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 1548.50 | 1552.00 | 1547.17 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1584.10 | 1552.00 | 1547.17 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1600.00 | 1581.54 | 1568.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1553.20 | 1576.04 | 1570.53 | SL hit (close<ema400) qty=1.00 sl=1570.53 alert=retest1 |

### Cycle 94 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 1560.40 | 1567.32 | 1567.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1551.00 | 1563.22 | 1565.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 1560.80 | 1559.78 | 1563.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 15:15:00 | 1560.80 | 1559.78 | 1563.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1560.80 | 1559.78 | 1563.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 1560.10 | 1559.78 | 1563.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1558.50 | 1559.53 | 1562.74 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1571.80 | 1564.55 | 1563.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1606.70 | 1572.98 | 1567.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 1567.50 | 1575.44 | 1570.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 1567.50 | 1575.44 | 1570.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1567.50 | 1575.44 | 1570.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1566.80 | 1575.44 | 1570.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 1564.40 | 1573.23 | 1569.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 1570.00 | 1573.23 | 1569.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 15:15:00 | 1558.00 | 1569.14 | 1568.60 | SL hit (close<static) qty=1.00 sl=1560.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 1563.70 | 1568.05 | 1568.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 1532.00 | 1552.71 | 1559.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1541.00 | 1532.51 | 1542.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1541.00 | 1532.51 | 1542.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1541.00 | 1532.51 | 1542.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1541.00 | 1532.51 | 1542.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1528.30 | 1531.66 | 1541.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1519.00 | 1531.66 | 1541.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1546.00 | 1534.22 | 1538.59 | SL hit (close>static) qty=1.00 sl=1542.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 1551.20 | 1541.37 | 1540.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1575.30 | 1550.08 | 1545.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1595.30 | 1597.18 | 1582.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:30:00 | 1599.30 | 1597.18 | 1582.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1589.10 | 1593.89 | 1587.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 1611.10 | 1595.44 | 1589.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1610.10 | 1598.98 | 1593.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:00:00 | 1625.10 | 1607.10 | 1599.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-03 09:15:00 | 1772.21 | 1730.29 | 1706.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 1622.80 | 1698.48 | 1707.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 1613.30 | 1681.44 | 1699.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 1609.40 | 1608.22 | 1630.78 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 11:45:00 | 1603.00 | 1607.15 | 1628.25 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 13:00:00 | 1603.10 | 1606.34 | 1625.96 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1606.00 | 1607.54 | 1620.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 1618.80 | 1607.54 | 1620.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1624.00 | 1580.84 | 1584.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1624.00 | 1580.84 | 1584.26 | SL hit (close>ema400) qty=1.00 sl=1584.26 alert=retest1 |

### Cycle 99 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1620.00 | 1588.67 | 1587.51 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 1582.40 | 1590.35 | 1591.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1572.80 | 1584.88 | 1588.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 10:15:00 | 1585.70 | 1585.04 | 1588.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 10:15:00 | 1585.70 | 1585.04 | 1588.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1585.70 | 1585.04 | 1588.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 1585.70 | 1585.04 | 1588.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1593.60 | 1586.76 | 1588.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:30:00 | 1593.30 | 1586.76 | 1588.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1595.50 | 1588.50 | 1589.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 1596.90 | 1588.50 | 1589.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 13:15:00 | 1599.30 | 1590.66 | 1590.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 15:15:00 | 1606.50 | 1595.80 | 1592.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 1593.00 | 1595.24 | 1592.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 1593.00 | 1595.24 | 1592.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1593.00 | 1595.24 | 1592.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 1593.00 | 1595.24 | 1592.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1585.00 | 1593.20 | 1592.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:45:00 | 1584.00 | 1593.20 | 1592.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1584.60 | 1591.48 | 1591.35 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 1583.00 | 1589.78 | 1590.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 1579.90 | 1585.93 | 1588.45 | Break + close below crossover candle low |

### Cycle 103 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 1632.00 | 1595.15 | 1592.41 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 1598.50 | 1603.72 | 1603.81 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 1608.30 | 1604.08 | 1603.82 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 1601.60 | 1603.58 | 1603.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 1596.10 | 1601.99 | 1602.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 1607.00 | 1601.05 | 1602.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 1607.00 | 1601.05 | 1602.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1607.00 | 1601.05 | 1602.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 1608.00 | 1601.05 | 1602.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1607.90 | 1602.42 | 1602.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 1608.00 | 1602.42 | 1602.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 1596.60 | 1601.25 | 1602.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:45:00 | 1600.90 | 1601.25 | 1602.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 1601.60 | 1601.32 | 1602.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 1599.50 | 1601.32 | 1602.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1600.40 | 1601.14 | 1601.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 1603.00 | 1601.14 | 1601.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1529.70 | 1520.65 | 1541.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 1529.70 | 1520.65 | 1541.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1571.50 | 1530.82 | 1544.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 1571.50 | 1530.82 | 1544.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1575.00 | 1539.65 | 1547.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:15:00 | 1582.90 | 1539.65 | 1547.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 1585.80 | 1555.79 | 1553.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 1610.00 | 1566.63 | 1558.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 12:15:00 | 1581.90 | 1592.79 | 1577.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 13:00:00 | 1581.90 | 1592.79 | 1577.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1574.20 | 1589.08 | 1577.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:45:00 | 1574.70 | 1589.08 | 1577.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 1565.50 | 1584.36 | 1576.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 1565.50 | 1584.36 | 1576.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1578.70 | 1577.88 | 1574.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 1590.10 | 1579.70 | 1575.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:15:00 | 1590.20 | 1579.70 | 1575.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 12:15:00 | 1568.20 | 1580.68 | 1580.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 1568.20 | 1580.68 | 1580.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 1555.00 | 1571.04 | 1576.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 1531.10 | 1529.23 | 1540.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 15:00:00 | 1531.10 | 1529.23 | 1540.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1488.70 | 1506.69 | 1520.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 1484.40 | 1498.66 | 1513.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:45:00 | 1485.30 | 1495.85 | 1510.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 1481.00 | 1492.84 | 1507.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1410.18 | 1478.92 | 1498.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1411.03 | 1478.92 | 1498.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1474.20 | 1468.70 | 1485.06 | SL hit (close>ema200) qty=0.50 sl=1468.70 alert=retest2 |

### Cycle 109 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 1461.10 | 1454.59 | 1454.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 1468.40 | 1459.78 | 1456.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 10:15:00 | 1522.60 | 1528.40 | 1516.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 1522.60 | 1528.40 | 1516.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1516.40 | 1526.00 | 1516.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1516.40 | 1526.00 | 1516.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1516.00 | 1524.00 | 1516.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 1516.00 | 1524.00 | 1516.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1513.70 | 1521.94 | 1516.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 1513.70 | 1521.94 | 1516.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1503.00 | 1518.15 | 1515.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 1503.00 | 1518.15 | 1515.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1467.20 | 1505.09 | 1509.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 1443.50 | 1492.77 | 1503.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 1404.70 | 1400.89 | 1419.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 13:45:00 | 1405.00 | 1400.89 | 1419.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1430.00 | 1409.03 | 1418.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 1430.00 | 1409.03 | 1418.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1449.70 | 1417.17 | 1421.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 1449.70 | 1417.17 | 1421.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 1433.20 | 1423.87 | 1423.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 15:15:00 | 1446.10 | 1432.03 | 1427.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 1446.60 | 1450.63 | 1440.68 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 15:15:00 | 1458.00 | 1449.69 | 1441.15 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1458.00 | 1451.35 | 1442.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1440.00 | 1451.35 | 1442.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1438.20 | 1448.72 | 1442.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1438.20 | 1448.72 | 1442.28 | SL hit (close<ema400) qty=1.00 sl=1442.28 alert=retest1 |

### Cycle 112 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 1428.00 | 1441.12 | 1441.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 1424.00 | 1437.70 | 1439.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 1404.80 | 1402.36 | 1411.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 10:15:00 | 1404.80 | 1402.36 | 1411.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1404.80 | 1402.36 | 1411.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 1404.80 | 1402.36 | 1411.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1414.80 | 1405.62 | 1411.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 1415.00 | 1405.62 | 1411.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1419.30 | 1408.36 | 1412.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:45:00 | 1410.30 | 1409.00 | 1412.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 1420.10 | 1413.94 | 1413.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1420.10 | 1413.94 | 1413.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 1432.00 | 1420.71 | 1417.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 1403.30 | 1422.68 | 1420.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1403.30 | 1422.68 | 1420.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1403.30 | 1422.68 | 1420.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 1403.30 | 1422.68 | 1420.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 1401.70 | 1418.48 | 1419.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 15:15:00 | 1400.00 | 1407.58 | 1412.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1413.30 | 1408.72 | 1412.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 1413.30 | 1408.72 | 1412.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1413.30 | 1408.72 | 1412.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 1413.30 | 1408.72 | 1412.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1412.10 | 1409.40 | 1412.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 1404.60 | 1408.76 | 1412.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:00:00 | 1405.30 | 1408.07 | 1411.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:30:00 | 1409.10 | 1408.94 | 1411.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1426.20 | 1414.20 | 1413.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 1426.20 | 1414.20 | 1413.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1434.90 | 1422.74 | 1417.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 1450.00 | 1453.86 | 1440.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 09:45:00 | 1454.40 | 1453.86 | 1440.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1441.50 | 1451.39 | 1440.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 1441.50 | 1451.39 | 1440.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1437.60 | 1448.63 | 1440.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 1438.30 | 1448.63 | 1440.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1427.50 | 1444.40 | 1438.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1427.50 | 1444.40 | 1438.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1454.50 | 1448.44 | 1442.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1445.70 | 1448.44 | 1442.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1450.00 | 1448.75 | 1443.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 1450.00 | 1448.75 | 1443.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1451.70 | 1449.34 | 1444.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:45:00 | 1450.70 | 1449.34 | 1444.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1458.00 | 1451.75 | 1446.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 1469.00 | 1455.04 | 1449.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:15:00 | 1469.00 | 1454.52 | 1452.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1502.80 | 1528.78 | 1532.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 1502.80 | 1528.78 | 1532.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 15:15:00 | 1498.00 | 1513.83 | 1523.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 1520.00 | 1515.06 | 1522.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 1520.00 | 1515.06 | 1522.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1520.00 | 1515.06 | 1522.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 1520.00 | 1515.06 | 1522.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1507.40 | 1504.85 | 1512.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 1512.50 | 1504.85 | 1512.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1491.30 | 1502.14 | 1510.36 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1547.40 | 1517.38 | 1515.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1559.50 | 1535.59 | 1524.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 1558.50 | 1559.36 | 1546.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:45:00 | 1559.40 | 1559.36 | 1546.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1546.80 | 1556.37 | 1546.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:00:00 | 1546.80 | 1556.37 | 1546.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1553.80 | 1555.86 | 1547.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 1560.20 | 1554.49 | 1547.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1539.00 | 1550.34 | 1546.92 | SL hit (close<static) qty=1.00 sl=1545.60 alert=retest2 |

### Cycle 118 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 1530.90 | 1544.59 | 1544.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1526.00 | 1538.12 | 1540.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 1435.00 | 1426.05 | 1442.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 1437.10 | 1428.26 | 1442.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1437.10 | 1428.26 | 1442.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 1446.90 | 1428.26 | 1442.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1442.00 | 1433.23 | 1440.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 1442.00 | 1433.23 | 1440.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1442.00 | 1434.99 | 1440.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 1437.20 | 1434.99 | 1440.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1436.00 | 1435.19 | 1439.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:30:00 | 1433.30 | 1435.39 | 1439.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:00:00 | 1431.20 | 1435.05 | 1438.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 1420.00 | 1435.56 | 1438.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 1428.20 | 1429.94 | 1432.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1455.00 | 1431.60 | 1431.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 12:15:00 | 1455.00 | 1431.60 | 1431.86 | SL hit (close>static) qty=1.00 sl=1449.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 1441.00 | 1433.48 | 1432.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 1456.90 | 1447.72 | 1442.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 1497.60 | 1504.08 | 1485.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:00:00 | 1497.60 | 1504.08 | 1485.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1487.90 | 1499.94 | 1491.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 1486.30 | 1499.94 | 1491.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1490.20 | 1497.99 | 1491.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1511.20 | 1489.86 | 1489.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:15:00 | 1500.40 | 1496.92 | 1495.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 1485.00 | 1492.91 | 1493.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 13:15:00 | 1485.00 | 1492.91 | 1493.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 14:15:00 | 1482.30 | 1490.79 | 1492.79 | Break + close below crossover candle low |

### Cycle 121 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 1512.90 | 1494.28 | 1493.97 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1438.20 | 1486.84 | 1491.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 1405.80 | 1426.49 | 1443.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 11:15:00 | 1379.30 | 1375.80 | 1391.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:00:00 | 1379.30 | 1375.80 | 1391.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1393.00 | 1381.95 | 1390.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 1393.00 | 1381.95 | 1390.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 1392.80 | 1384.12 | 1390.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 1405.80 | 1384.12 | 1390.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 1429.00 | 1398.35 | 1396.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1448.50 | 1426.00 | 1417.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1423.10 | 1429.98 | 1421.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 1423.10 | 1429.98 | 1421.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1420.20 | 1428.02 | 1421.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:00:00 | 1420.20 | 1428.02 | 1421.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 1418.00 | 1426.02 | 1421.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:30:00 | 1420.00 | 1426.02 | 1421.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 1413.10 | 1423.43 | 1420.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 1412.50 | 1423.43 | 1420.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 1411.50 | 1417.93 | 1418.43 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 1431.00 | 1420.55 | 1419.58 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1415.70 | 1431.43 | 1433.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 1378.50 | 1414.25 | 1423.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 1375.50 | 1375.40 | 1389.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 1375.50 | 1375.40 | 1389.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1386.70 | 1376.24 | 1381.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 1386.70 | 1376.24 | 1381.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 1394.90 | 1379.98 | 1383.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 1391.50 | 1379.98 | 1383.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1377.40 | 1379.93 | 1382.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 1375.80 | 1379.11 | 1381.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:00:00 | 1374.70 | 1378.23 | 1381.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1370.90 | 1379.08 | 1380.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 1375.80 | 1377.82 | 1380.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 1379.10 | 1377.79 | 1379.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:00:00 | 1369.70 | 1375.88 | 1378.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1307.01 | 1331.34 | 1346.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1305.96 | 1331.34 | 1346.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1302.36 | 1331.34 | 1346.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1307.01 | 1331.34 | 1346.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1301.21 | 1331.34 | 1346.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 1323.00 | 1320.16 | 1332.04 | SL hit (close>ema200) qty=0.50 sl=1320.16 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 1247.10 | 1236.58 | 1235.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 1253.70 | 1242.38 | 1239.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 1248.10 | 1248.61 | 1243.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:00:00 | 1248.10 | 1248.61 | 1243.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1243.00 | 1247.49 | 1243.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 1243.00 | 1247.49 | 1243.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1241.50 | 1246.29 | 1243.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 1241.50 | 1246.29 | 1243.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1232.80 | 1243.59 | 1242.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 1232.80 | 1243.59 | 1242.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1226.20 | 1240.11 | 1240.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 1218.40 | 1230.75 | 1235.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 1201.00 | 1196.45 | 1205.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:00:00 | 1201.00 | 1196.45 | 1205.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1206.50 | 1198.46 | 1205.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1206.50 | 1198.46 | 1205.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1210.00 | 1200.77 | 1205.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1230.10 | 1200.77 | 1205.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1234.30 | 1207.47 | 1208.56 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1233.20 | 1212.62 | 1210.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 1236.20 | 1222.16 | 1216.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 1253.90 | 1259.48 | 1251.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 1253.90 | 1259.48 | 1251.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1258.90 | 1258.73 | 1252.52 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1239.60 | 1249.72 | 1250.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 1211.50 | 1237.91 | 1244.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1263.90 | 1228.90 | 1234.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 1263.90 | 1228.90 | 1234.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1263.90 | 1228.90 | 1234.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 1275.00 | 1228.90 | 1234.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 1282.40 | 1239.60 | 1238.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 1292.20 | 1257.77 | 1247.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 1267.40 | 1271.97 | 1258.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 1267.40 | 1271.97 | 1258.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1245.60 | 1266.70 | 1257.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 1245.60 | 1266.70 | 1257.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1241.20 | 1261.60 | 1256.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 1240.80 | 1261.60 | 1256.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 1243.60 | 1252.79 | 1253.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1242.00 | 1249.38 | 1251.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 1262.10 | 1251.93 | 1252.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 1262.10 | 1251.93 | 1252.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1262.10 | 1251.93 | 1252.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 1262.10 | 1251.93 | 1252.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1268.50 | 1255.24 | 1253.79 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 1245.80 | 1254.95 | 1255.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 1240.00 | 1251.54 | 1253.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 1243.20 | 1240.41 | 1244.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:00:00 | 1243.20 | 1240.41 | 1244.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1217.80 | 1222.25 | 1230.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 1224.90 | 1222.25 | 1230.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1208.50 | 1202.02 | 1209.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1213.60 | 1202.02 | 1209.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1224.50 | 1206.51 | 1211.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 1224.50 | 1206.51 | 1211.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1229.60 | 1211.13 | 1212.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1223.20 | 1213.54 | 1213.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 1221.20 | 1215.07 | 1214.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 1221.20 | 1215.07 | 1214.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 15:15:00 | 1225.10 | 1219.04 | 1216.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 13:15:00 | 1221.20 | 1222.07 | 1219.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 14:00:00 | 1221.20 | 1222.07 | 1219.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1216.50 | 1220.96 | 1219.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 15:00:00 | 1216.50 | 1220.96 | 1219.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1222.00 | 1221.17 | 1219.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:00:00 | 1226.90 | 1222.31 | 1219.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 10:15:00 | 1206.60 | 1219.17 | 1218.78 | SL hit (close<static) qty=1.00 sl=1214.70 alert=retest2 |

### Cycle 136 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 1207.80 | 1216.90 | 1217.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 1198.40 | 1211.65 | 1215.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1155.00 | 1152.08 | 1165.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 1174.00 | 1152.08 | 1165.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1173.60 | 1156.38 | 1166.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1171.10 | 1156.38 | 1166.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1165.40 | 1158.19 | 1166.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 1162.10 | 1159.39 | 1166.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 1179.00 | 1163.31 | 1167.30 | SL hit (close>static) qty=1.00 sl=1174.90 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1183.70 | 1172.40 | 1170.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 1191.00 | 1176.12 | 1172.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 1179.40 | 1179.81 | 1175.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:00:00 | 1179.40 | 1179.81 | 1175.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1180.80 | 1180.01 | 1176.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:15:00 | 1172.10 | 1180.01 | 1176.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1172.10 | 1178.43 | 1175.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1166.10 | 1178.43 | 1175.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1176.20 | 1177.98 | 1175.96 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 1168.80 | 1174.07 | 1174.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1155.40 | 1168.92 | 1171.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 1169.80 | 1166.71 | 1170.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 1170.70 | 1166.71 | 1170.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1165.20 | 1166.40 | 1169.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 11:30:00 | 1157.90 | 1164.34 | 1168.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:30:00 | 1161.00 | 1165.07 | 1168.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 1175.00 | 1168.02 | 1168.92 | SL hit (close>static) qty=1.00 sl=1173.10 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 1041.20 | 1021.57 | 1020.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 1042.90 | 1028.43 | 1024.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1029.60 | 1033.44 | 1027.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1029.60 | 1033.44 | 1027.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1029.60 | 1033.44 | 1027.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 1029.60 | 1033.44 | 1027.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1025.30 | 1031.81 | 1027.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 1025.30 | 1031.81 | 1027.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1020.70 | 1029.59 | 1027.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:45:00 | 1023.20 | 1029.59 | 1027.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 14:15:00 | 1017.00 | 1024.37 | 1025.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 986.40 | 1015.87 | 1021.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1038.20 | 1008.80 | 1012.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1038.20 | 1008.80 | 1012.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1038.20 | 1008.80 | 1012.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1038.20 | 1008.80 | 1012.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1063.20 | 1019.68 | 1017.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 1073.40 | 1038.32 | 1026.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1063.00 | 1066.85 | 1055.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:30:00 | 1058.90 | 1066.85 | 1055.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1059.00 | 1065.17 | 1056.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 1059.30 | 1065.17 | 1056.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1056.90 | 1063.52 | 1056.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:45:00 | 1057.30 | 1063.52 | 1056.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1062.00 | 1063.22 | 1057.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 1064.90 | 1062.87 | 1057.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1050.80 | 1060.78 | 1057.53 | SL hit (close<static) qty=1.00 sl=1055.10 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 1049.50 | 1054.48 | 1055.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 1044.00 | 1050.70 | 1053.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 1038.60 | 1028.38 | 1034.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 1038.60 | 1028.38 | 1034.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1038.60 | 1028.38 | 1034.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1044.00 | 1028.38 | 1034.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1038.90 | 1030.48 | 1034.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1043.80 | 1030.48 | 1034.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1034.10 | 1033.56 | 1035.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:15:00 | 1033.30 | 1033.56 | 1035.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:45:00 | 1033.10 | 1033.73 | 1035.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 1049.00 | 1036.89 | 1036.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1049.00 | 1036.89 | 1036.47 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 15:15:00 | 1030.00 | 1035.64 | 1036.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 1020.30 | 1032.57 | 1034.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 1001.50 | 1001.09 | 1007.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 15:00:00 | 1001.50 | 1001.09 | 1007.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 996.60 | 993.25 | 997.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 991.50 | 993.45 | 996.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 990.00 | 991.46 | 994.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 13:00:00 | 989.60 | 985.28 | 988.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 14:15:00 | 992.10 | 987.09 | 989.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1005.00 | 993.02 | 991.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 09:15:00 | 1005.00 | 993.02 | 991.66 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 981.00 | 990.13 | 990.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 975.55 | 987.21 | 989.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 987.00 | 984.48 | 987.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 952.70 | 984.48 | 987.18 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 946.15 | 951.37 | 964.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 936.50 | 945.53 | 957.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 965.70 | 950.97 | 957.29 | SL hit (close>ema400) qty=1.00 sl=957.29 alert=retest1 |

### Cycle 147 — BUY (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 15:15:00 | 970.00 | 959.22 | 958.95 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 915.85 | 950.55 | 955.03 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 985.60 | 949.22 | 949.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1017.35 | 979.69 | 966.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 996.00 | 996.22 | 980.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 996.00 | 996.22 | 980.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 976.10 | 993.34 | 982.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 972.00 | 993.34 | 982.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 987.95 | 992.26 | 982.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 991.00 | 992.41 | 983.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 992.30 | 992.39 | 984.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 953.70 | 980.97 | 981.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 953.70 | 980.97 | 981.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 945.30 | 966.15 | 973.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 11:15:00 | 926.80 | 919.57 | 929.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 12:00:00 | 926.80 | 919.57 | 929.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 943.00 | 924.26 | 930.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 943.00 | 924.26 | 930.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 945.00 | 928.40 | 932.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:15:00 | 950.80 | 928.40 | 932.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 931.00 | 928.52 | 930.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:45:00 | 931.85 | 928.52 | 930.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 922.80 | 927.37 | 930.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 930.45 | 927.37 | 930.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 921.20 | 922.84 | 927.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 918.65 | 922.84 | 927.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 915.85 | 922.22 | 926.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 918.95 | 921.61 | 925.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 919.25 | 920.70 | 924.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 872.72 | 895.33 | 909.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 873.00 | 895.33 | 909.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 873.29 | 895.33 | 909.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 15:15:00 | 870.06 | 886.03 | 901.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 877.00 | 884.22 | 898.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 873.40 | 882.82 | 896.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 897.70 | 885.79 | 897.01 | SL hit (close>ema200) qty=0.50 sl=885.79 alert=retest2 |

### Cycle 151 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 918.85 | 903.40 | 903.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 977.20 | 920.89 | 911.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 977.20 | 979.87 | 953.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 977.20 | 979.87 | 953.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 955.35 | 967.24 | 956.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 955.35 | 967.24 | 956.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 948.50 | 963.49 | 955.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 933.35 | 963.49 | 955.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 910.85 | 946.32 | 948.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 909.55 | 928.26 | 938.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 971.85 | 934.37 | 939.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 971.85 | 934.37 | 939.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 971.85 | 934.37 | 939.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 960.55 | 934.37 | 939.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 968.05 | 941.11 | 941.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 969.90 | 941.11 | 941.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 987.20 | 950.33 | 946.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 992.40 | 958.74 | 950.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 956.10 | 969.41 | 959.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 956.10 | 969.41 | 959.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 956.10 | 969.41 | 959.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 969.00 | 967.90 | 959.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 970.90 | 978.81 | 969.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 1065.90 | 1030.93 | 1009.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1223.55 | 1248.45 | 1250.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1213.75 | 1237.76 | 1244.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1245.60 | 1233.45 | 1239.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1245.60 | 1233.45 | 1239.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1245.60 | 1233.45 | 1239.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1249.50 | 1233.45 | 1239.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1243.75 | 1235.51 | 1240.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1251.30 | 1235.51 | 1240.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1245.60 | 1238.85 | 1241.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 1245.60 | 1238.85 | 1241.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 1240.80 | 1239.47 | 1240.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:45:00 | 1243.10 | 1239.47 | 1240.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 1239.00 | 1239.38 | 1240.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 1237.15 | 1239.38 | 1240.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1234.90 | 1238.48 | 1240.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 1228.45 | 1237.22 | 1239.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 1226.70 | 1237.22 | 1239.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 1244.90 | 1236.45 | 1236.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 1244.90 | 1236.45 | 1236.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 11:15:00 | 1265.90 | 1245.39 | 1240.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 15:15:00 | 1242.75 | 1254.78 | 1247.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 1242.75 | 1254.78 | 1247.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1242.75 | 1254.78 | 1247.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1278.00 | 1254.78 | 1247.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1275.50 | 1272.62 | 1269.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:15:00 | 1277.20 | 1270.61 | 1268.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-03 09:15:00 | 570.30 | 2024-06-04 09:15:00 | 550.45 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-06-03 11:30:00 | 565.40 | 2024-06-04 09:15:00 | 550.45 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-06-03 12:15:00 | 565.25 | 2024-06-04 09:15:00 | 550.45 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-06-13 12:30:00 | 599.65 | 2024-06-14 14:15:00 | 593.25 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-06-13 13:00:00 | 600.95 | 2024-06-14 14:15:00 | 593.25 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-06-14 09:30:00 | 600.00 | 2024-06-14 14:15:00 | 593.25 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-07-08 14:30:00 | 772.20 | 2024-07-10 09:15:00 | 746.55 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2024-07-08 15:00:00 | 772.00 | 2024-07-10 09:15:00 | 746.55 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-07-09 09:15:00 | 788.00 | 2024-07-10 09:15:00 | 746.55 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2024-07-18 15:15:00 | 755.05 | 2024-07-22 10:15:00 | 786.00 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2024-07-19 10:00:00 | 745.05 | 2024-07-22 10:15:00 | 786.00 | STOP_HIT | 1.00 | -5.50% |
| SELL | retest2 | 2024-07-19 15:15:00 | 745.00 | 2024-07-22 10:15:00 | 786.00 | STOP_HIT | 1.00 | -5.50% |
| BUY | retest2 | 2024-08-01 09:30:00 | 959.10 | 2024-08-05 11:15:00 | 921.70 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2024-08-02 09:15:00 | 956.55 | 2024-08-05 11:15:00 | 921.70 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2024-08-02 10:30:00 | 943.25 | 2024-08-05 11:15:00 | 921.70 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-08-02 12:15:00 | 954.75 | 2024-08-05 11:15:00 | 921.70 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1002.80 | 2024-08-13 12:15:00 | 976.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-08-09 13:45:00 | 990.90 | 2024-08-13 12:15:00 | 976.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-09 14:45:00 | 992.85 | 2024-08-13 12:15:00 | 976.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-08-12 10:00:00 | 993.25 | 2024-08-13 12:15:00 | 976.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-08-20 11:15:00 | 945.75 | 2024-08-21 09:15:00 | 994.30 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest2 | 2024-08-28 09:15:00 | 1085.00 | 2024-08-28 11:15:00 | 1070.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-08-30 12:15:00 | 1051.90 | 2024-09-02 09:15:00 | 1085.00 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-08-30 15:00:00 | 1044.70 | 2024-09-02 09:15:00 | 1085.00 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2024-09-27 15:00:00 | 1059.05 | 2024-10-07 09:15:00 | 1039.35 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-10-25 15:15:00 | 1155.00 | 2024-10-29 15:15:00 | 1270.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-12 13:45:00 | 1299.85 | 2024-11-13 10:15:00 | 1234.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 13:45:00 | 1299.85 | 2024-11-14 09:15:00 | 1285.20 | STOP_HIT | 0.50 | 1.13% |
| SELL | retest2 | 2024-11-14 12:30:00 | 1297.50 | 2024-11-18 10:15:00 | 1315.95 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-11-14 13:30:00 | 1304.20 | 2024-11-18 10:15:00 | 1315.95 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-11-27 09:15:00 | 1313.70 | 2024-12-04 13:15:00 | 1355.00 | STOP_HIT | 1.00 | 3.14% |
| BUY | retest2 | 2024-11-27 10:30:00 | 1315.50 | 2024-12-04 13:15:00 | 1355.00 | STOP_HIT | 1.00 | 3.00% |
| BUY | retest2 | 2024-12-10 11:30:00 | 1406.65 | 2024-12-10 15:15:00 | 1380.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-12-17 12:15:00 | 1305.95 | 2024-12-18 12:15:00 | 1240.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 12:15:00 | 1305.95 | 2024-12-19 10:15:00 | 1238.50 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2025-01-02 14:00:00 | 1223.05 | 2025-01-03 14:15:00 | 1201.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-01-02 15:15:00 | 1223.00 | 2025-01-03 14:15:00 | 1201.80 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-01-03 11:00:00 | 1219.25 | 2025-01-03 14:15:00 | 1201.80 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1168.65 | 2025-01-09 09:15:00 | 1208.25 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-02-05 11:15:00 | 1139.40 | 2025-02-06 09:15:00 | 1159.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-02-13 11:15:00 | 1080.75 | 2025-02-14 09:15:00 | 1026.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 1081.30 | 2025-02-14 09:15:00 | 1027.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:15:00 | 1080.75 | 2025-02-17 09:15:00 | 972.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 1081.30 | 2025-02-17 09:15:00 | 973.17 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-24 11:15:00 | 1069.85 | 2025-02-27 09:15:00 | 1044.60 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-02-25 09:15:00 | 1084.65 | 2025-02-27 09:15:00 | 1044.60 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest1 | 2025-02-28 09:15:00 | 972.35 | 2025-03-03 09:15:00 | 923.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-28 09:15:00 | 972.35 | 2025-03-04 09:15:00 | 981.00 | STOP_HIT | 0.50 | -0.89% |
| BUY | retest2 | 2025-03-17 09:30:00 | 1128.10 | 2025-03-17 13:15:00 | 1115.75 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-03-17 10:30:00 | 1131.10 | 2025-03-17 13:15:00 | 1115.75 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-03-18 12:15:00 | 1117.00 | 2025-03-21 09:15:00 | 1118.15 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-03-18 13:15:00 | 1112.35 | 2025-03-21 09:15:00 | 1118.15 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-03-20 09:30:00 | 1110.50 | 2025-03-21 09:15:00 | 1118.15 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-03-20 11:00:00 | 1116.90 | 2025-03-21 09:15:00 | 1118.15 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-03-25 09:15:00 | 1131.95 | 2025-03-26 13:15:00 | 1112.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-04-23 09:15:00 | 1295.40 | 2025-04-29 15:15:00 | 1302.00 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-04-23 10:15:00 | 1284.90 | 2025-04-29 15:15:00 | 1302.00 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2025-04-23 11:00:00 | 1284.20 | 2025-04-29 15:15:00 | 1302.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-04-23 15:00:00 | 1284.20 | 2025-04-29 15:15:00 | 1302.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-04-24 10:30:00 | 1300.40 | 2025-04-29 15:15:00 | 1302.00 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-04-25 12:15:00 | 1301.80 | 2025-04-29 15:15:00 | 1302.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-04-28 09:15:00 | 1301.90 | 2025-04-29 15:15:00 | 1302.00 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-04-29 09:45:00 | 1314.00 | 2025-04-29 15:15:00 | 1302.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-05-05 11:15:00 | 1278.80 | 2025-05-07 09:15:00 | 1214.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1269.90 | 2025-05-07 09:15:00 | 1206.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 11:15:00 | 1278.80 | 2025-05-07 11:15:00 | 1259.20 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1269.90 | 2025-05-07 11:15:00 | 1259.20 | STOP_HIT | 0.50 | 0.84% |
| BUY | retest2 | 2025-06-03 09:15:00 | 1534.50 | 2025-06-09 12:15:00 | 1519.70 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest1 | 2025-06-11 09:15:00 | 1584.10 | 2025-06-12 13:15:00 | 1553.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-17 14:15:00 | 1570.00 | 2025-06-17 15:15:00 | 1558.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1519.00 | 2025-06-23 09:15:00 | 1546.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-06-27 13:15:00 | 1611.10 | 2025-07-03 09:15:00 | 1772.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1610.10 | 2025-07-03 09:15:00 | 1771.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 13:00:00 | 1625.10 | 2025-07-07 09:15:00 | 1622.80 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-07-09 11:45:00 | 1603.00 | 2025-07-15 09:15:00 | 1624.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest1 | 2025-07-09 13:00:00 | 1603.10 | 2025-07-15 09:15:00 | 1624.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-31 11:30:00 | 1590.10 | 2025-08-01 12:15:00 | 1568.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-31 12:15:00 | 1590.20 | 2025-08-01 12:15:00 | 1568.20 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-08-08 13:00:00 | 1484.40 | 2025-08-11 09:15:00 | 1410.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 13:45:00 | 1485.30 | 2025-08-11 09:15:00 | 1411.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 13:00:00 | 1484.40 | 2025-08-11 14:15:00 | 1474.20 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2025-08-08 13:45:00 | 1485.30 | 2025-08-11 14:15:00 | 1474.20 | STOP_HIT | 0.50 | 0.75% |
| SELL | retest2 | 2025-08-08 14:45:00 | 1481.00 | 2025-08-18 12:15:00 | 1461.10 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest1 | 2025-09-03 15:15:00 | 1458.00 | 2025-09-04 09:15:00 | 1438.20 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-09-04 13:45:00 | 1453.30 | 2025-09-05 09:15:00 | 1428.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-09 14:45:00 | 1410.30 | 2025-09-10 10:15:00 | 1420.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-09-15 11:30:00 | 1404.60 | 2025-09-16 11:15:00 | 1426.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-09-15 13:00:00 | 1405.30 | 2025-09-16 11:15:00 | 1426.20 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-09-15 14:30:00 | 1409.10 | 2025-09-16 11:15:00 | 1426.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-09-22 09:30:00 | 1469.00 | 2025-09-29 11:15:00 | 1502.80 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest2 | 2025-09-23 12:15:00 | 1469.00 | 2025-09-29 11:15:00 | 1502.80 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1560.20 | 2025-10-07 10:15:00 | 1539.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-17 10:30:00 | 1433.30 | 2025-10-23 12:15:00 | 1455.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-17 13:00:00 | 1431.20 | 2025-10-23 12:15:00 | 1455.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-10-20 10:45:00 | 1420.00 | 2025-10-23 12:15:00 | 1455.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-10-23 09:15:00 | 1428.20 | 2025-10-23 12:15:00 | 1455.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1511.20 | 2025-11-03 13:15:00 | 1485.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-11-03 10:15:00 | 1500.40 | 2025-11-03 13:15:00 | 1485.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1375.80 | 2025-12-03 09:15:00 | 1307.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 14:00:00 | 1374.70 | 2025-12-03 09:15:00 | 1305.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1370.90 | 2025-12-03 09:15:00 | 1302.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 11:15:00 | 1375.80 | 2025-12-03 09:15:00 | 1307.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 15:00:00 | 1369.70 | 2025-12-03 09:15:00 | 1301.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1375.80 | 2025-12-04 09:15:00 | 1323.00 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-11-27 14:00:00 | 1374.70 | 2025-12-04 09:15:00 | 1323.00 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1370.90 | 2025-12-04 09:15:00 | 1323.00 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2025-11-28 11:15:00 | 1375.80 | 2025-12-04 09:15:00 | 1323.00 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-11-28 15:00:00 | 1369.70 | 2025-12-04 09:15:00 | 1323.00 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1223.20 | 2026-01-13 12:15:00 | 1221.20 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2026-01-16 10:00:00 | 1226.90 | 2026-01-16 10:15:00 | 1206.60 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-01-22 11:45:00 | 1162.10 | 2026-01-22 12:15:00 | 1179.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-28 11:30:00 | 1157.90 | 2026-01-28 15:15:00 | 1175.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-28 12:30:00 | 1161.00 | 2026-01-28 15:15:00 | 1175.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-29 09:30:00 | 1152.00 | 2026-01-29 15:15:00 | 1094.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:30:00 | 1152.00 | 2026-01-30 09:15:00 | 1036.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-11 15:15:00 | 1064.90 | 2026-02-12 09:15:00 | 1050.80 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-17 14:15:00 | 1033.30 | 2026-02-18 09:15:00 | 1049.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-17 14:45:00 | 1033.10 | 2026-02-18 09:15:00 | 1049.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-02-25 12:45:00 | 991.50 | 2026-03-02 09:15:00 | 1005.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-02-26 11:30:00 | 990.00 | 2026-03-02 09:15:00 | 1005.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-27 13:00:00 | 989.60 | 2026-03-02 09:15:00 | 1005.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-02-27 14:15:00 | 992.10 | 2026-03-02 09:15:00 | 1005.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest1 | 2026-03-04 09:15:00 | 952.70 | 2026-03-06 09:15:00 | 965.70 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-03-05 13:45:00 | 936.50 | 2026-03-06 15:15:00 | 970.00 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2026-03-12 11:45:00 | 991.00 | 2026-03-13 09:15:00 | 953.70 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2026-03-12 13:00:00 | 992.30 | 2026-03-13 09:15:00 | 953.70 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2026-03-20 10:15:00 | 918.65 | 2026-03-23 12:15:00 | 872.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 915.85 | 2026-03-23 12:15:00 | 873.00 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2026-03-20 12:45:00 | 918.95 | 2026-03-23 12:15:00 | 873.29 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-03-20 13:45:00 | 919.25 | 2026-03-23 15:15:00 | 870.06 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2026-03-20 10:15:00 | 918.65 | 2026-03-24 11:15:00 | 897.70 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-03-20 12:15:00 | 915.85 | 2026-03-24 11:15:00 | 897.70 | STOP_HIT | 0.50 | 1.98% |
| SELL | retest2 | 2026-03-20 12:45:00 | 918.95 | 2026-03-24 11:15:00 | 897.70 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2026-03-20 13:45:00 | 919.25 | 2026-03-24 11:15:00 | 897.70 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2026-03-24 10:30:00 | 873.40 | 2026-03-24 12:15:00 | 918.60 | STOP_HIT | 1.00 | -5.18% |
| BUY | retest2 | 2026-04-02 11:15:00 | 969.00 | 2026-04-09 09:15:00 | 1065.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:15:00 | 970.90 | 2026-04-09 09:15:00 | 1067.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 10:30:00 | 1228.45 | 2026-04-29 12:15:00 | 1244.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-04-28 11:15:00 | 1226.70 | 2026-04-29 12:15:00 | 1244.90 | STOP_HIT | 1.00 | -1.48% |
