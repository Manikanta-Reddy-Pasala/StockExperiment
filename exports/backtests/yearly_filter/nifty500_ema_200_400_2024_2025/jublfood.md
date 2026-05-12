# Jubilant Foodworks Ltd. (JUBLFOOD)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 473.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 26
- **Target hits / Stop hits / Partials:** 0 / 29 / 3
- **Avg / median % per leg:** -2.13% / -1.90%
- **Sum % (uncompounded):** -68.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -2.87% | -43.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -2.87% | -43.0% |
| SELL (all) | 17 | 6 | 35.3% | 0 | 14 | 3 | -1.48% | -25.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 0 | 14 | 3 | -1.48% | -25.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 6 | 18.8% | 0 | 29 | 3 | -2.13% | -68.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 11:15:00 | 516.00 | 469.67 | 469.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 13:15:00 | 518.20 | 470.62 | 470.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 09:15:00 | 664.00 | 667.77 | 636.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 10:00:00 | 664.00 | 667.77 | 636.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 636.00 | 666.42 | 637.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:30:00 | 634.30 | 666.42 | 637.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 634.30 | 666.10 | 637.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 634.30 | 666.10 | 637.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 635.50 | 664.58 | 637.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:15:00 | 632.45 | 664.58 | 637.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 638.90 | 664.32 | 637.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:30:00 | 636.40 | 664.32 | 637.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 631.45 | 664.00 | 637.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 631.45 | 664.00 | 637.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 631.30 | 663.67 | 637.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:30:00 | 631.30 | 663.67 | 637.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 632.10 | 641.93 | 632.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:30:00 | 633.20 | 641.93 | 632.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 629.25 | 641.81 | 632.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:45:00 | 627.15 | 641.81 | 632.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 619.85 | 641.59 | 632.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 619.85 | 641.59 | 632.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 15:15:00 | 578.20 | 625.21 | 625.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 574.30 | 621.31 | 623.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 616.05 | 610.20 | 616.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 616.05 | 610.20 | 616.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 616.05 | 610.20 | 616.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 616.05 | 610.20 | 616.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 603.85 | 610.13 | 616.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 614.80 | 610.13 | 616.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 641.80 | 610.12 | 616.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:45:00 | 612.25 | 613.05 | 617.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:15:00 | 611.20 | 613.05 | 617.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:15:00 | 614.10 | 612.96 | 617.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 614.25 | 612.99 | 617.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 616.95 | 612.90 | 617.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 616.95 | 612.90 | 617.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 618.35 | 612.96 | 617.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 618.35 | 612.96 | 617.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 618.60 | 613.01 | 617.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:45:00 | 618.75 | 613.01 | 617.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 610.05 | 613.11 | 617.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-26 11:15:00 | 652.00 | 616.42 | 618.38 | SL hit (close>static) qty=1.00 sl=651.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 10:15:00 | 647.80 | 620.34 | 620.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 12:15:00 | 654.70 | 625.99 | 623.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 14:15:00 | 707.20 | 708.52 | 680.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-13 14:30:00 | 706.80 | 708.52 | 680.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 689.90 | 705.06 | 683.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 751.70 | 695.21 | 683.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:00:00 | 749.60 | 696.38 | 684.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 12:15:00 | 657.55 | 697.39 | 687.43 | SL hit (close<static) qty=1.00 sl=662.15 alert=retest2 |

### Cycle 4 — SELL (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 13:15:00 | 625.55 | 681.69 | 681.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 12:15:00 | 613.10 | 678.35 | 680.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 646.50 | 644.72 | 659.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 12:45:00 | 649.30 | 644.72 | 659.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 665.40 | 642.91 | 657.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:00:00 | 665.40 | 642.91 | 657.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 658.70 | 643.07 | 657.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:30:00 | 655.30 | 648.91 | 658.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 675.25 | 649.62 | 658.45 | SL hit (close>static) qty=1.00 sl=665.30 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 702.40 | 664.86 | 664.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 13:15:00 | 705.30 | 671.77 | 668.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 676.10 | 690.70 | 680.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 676.10 | 690.70 | 680.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 676.10 | 690.70 | 680.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 676.10 | 690.70 | 680.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 675.15 | 690.55 | 680.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 675.15 | 690.55 | 680.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 666.75 | 690.31 | 680.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 662.50 | 690.31 | 680.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 693.25 | 690.28 | 681.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 691.70 | 690.28 | 681.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 679.30 | 690.19 | 681.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 677.50 | 690.19 | 681.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 679.25 | 690.08 | 681.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 679.25 | 690.08 | 681.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 679.15 | 689.98 | 681.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 675.45 | 689.98 | 681.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 682.80 | 689.70 | 681.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 15:15:00 | 684.90 | 689.70 | 681.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 09:15:00 | 673.15 | 689.49 | 681.76 | SL hit (close<static) qty=1.00 sl=679.20 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 660.20 | 677.25 | 677.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 656.05 | 676.52 | 676.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 678.00 | 675.51 | 676.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 678.00 | 675.51 | 676.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 678.00 | 675.51 | 676.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 678.00 | 675.51 | 676.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 678.90 | 675.55 | 676.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:45:00 | 679.80 | 675.55 | 676.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 686.45 | 675.65 | 676.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 686.45 | 675.65 | 676.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 695.75 | 677.42 | 677.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 698.45 | 678.51 | 677.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 677.45 | 680.71 | 679.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 677.45 | 680.71 | 679.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 677.45 | 680.71 | 679.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 677.45 | 680.71 | 679.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 671.80 | 680.62 | 679.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 671.80 | 680.62 | 679.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 677.45 | 680.04 | 678.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:00:00 | 677.45 | 680.04 | 678.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 681.25 | 680.05 | 678.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 685.70 | 680.12 | 678.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:00:00 | 685.50 | 680.17 | 678.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:30:00 | 685.65 | 680.80 | 679.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 687.10 | 680.80 | 679.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 677.75 | 680.83 | 679.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:30:00 | 679.00 | 680.83 | 679.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 677.00 | 680.80 | 679.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 682.40 | 680.80 | 679.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 681.50 | 680.83 | 679.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 682.80 | 680.83 | 679.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 679.90 | 680.82 | 679.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:45:00 | 679.10 | 680.82 | 679.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 680.60 | 680.82 | 679.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 679.60 | 680.82 | 679.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 688.35 | 694.08 | 687.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:45:00 | 687.75 | 694.08 | 687.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 682.90 | 693.97 | 687.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 682.90 | 693.97 | 687.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 685.90 | 693.89 | 687.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 688.20 | 693.47 | 687.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 689.15 | 693.26 | 687.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 688.65 | 693.20 | 687.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 680.75 | 692.49 | 687.52 | SL hit (close<static) qty=1.00 sl=682.15 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 656.45 | 684.49 | 684.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 655.50 | 683.67 | 684.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 649.00 | 647.44 | 660.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 649.00 | 647.44 | 660.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 654.55 | 643.42 | 655.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 639.00 | 648.24 | 655.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 639.00 | 648.15 | 655.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:00:00 | 639.40 | 648.06 | 655.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.05 | 636.41 | 646.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.05 | 636.41 | 646.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.43 | 636.41 | 646.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 632.15 | 631.69 | 642.57 | SL hit (close>ema200) qty=0.50 sl=631.69 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-28 09:45:00 | 502.50 | 2024-05-29 11:15:00 | 516.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-05-28 11:45:00 | 502.75 | 2024-05-29 11:15:00 | 516.00 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-05-28 13:30:00 | 502.95 | 2024-05-29 11:15:00 | 516.00 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-11-14 11:45:00 | 612.25 | 2024-11-26 11:15:00 | 652.00 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2024-11-14 12:15:00 | 611.20 | 2024-11-26 11:15:00 | 652.00 | STOP_HIT | 1.00 | -6.68% |
| SELL | retest2 | 2024-11-18 11:15:00 | 614.10 | 2024-11-26 11:15:00 | 652.00 | STOP_HIT | 1.00 | -6.17% |
| SELL | retest2 | 2024-11-18 12:45:00 | 614.25 | 2024-11-26 11:15:00 | 652.00 | STOP_HIT | 1.00 | -6.15% |
| BUY | retest2 | 2025-02-03 10:45:00 | 751.70 | 2025-02-11 12:15:00 | 657.55 | STOP_HIT | 1.00 | -12.52% |
| BUY | retest2 | 2025-02-03 13:00:00 | 749.60 | 2025-02-11 12:15:00 | 657.55 | STOP_HIT | 1.00 | -12.28% |
| SELL | retest2 | 2025-04-01 10:30:00 | 655.30 | 2025-04-02 09:15:00 | 675.25 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-04-07 10:00:00 | 655.30 | 2025-04-07 14:15:00 | 669.80 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-04-07 10:45:00 | 655.65 | 2025-04-07 14:15:00 | 669.80 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-04-07 11:15:00 | 652.55 | 2025-04-07 14:15:00 | 669.80 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-05-15 15:15:00 | 684.90 | 2025-05-16 09:15:00 | 673.15 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-05-19 10:45:00 | 685.20 | 2025-05-20 13:15:00 | 675.60 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-05-19 11:45:00 | 685.45 | 2025-05-20 13:15:00 | 675.60 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-06-16 10:15:00 | 685.70 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-16 11:00:00 | 685.50 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-06-17 09:30:00 | 685.65 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-06-17 10:15:00 | 687.10 | 2025-07-11 09:15:00 | 679.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-08 11:00:00 | 688.20 | 2025-07-11 15:15:00 | 675.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-08 15:00:00 | 689.15 | 2025-07-11 15:15:00 | 675.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-07-09 09:15:00 | 688.65 | 2025-07-11 15:15:00 | 675.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-10 14:15:00 | 688.10 | 2025-07-11 15:15:00 | 675.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-07-16 09:15:00 | 692.10 | 2025-07-18 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-16 12:45:00 | 692.50 | 2025-07-18 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-12 10:15:00 | 639.00 | 2025-09-26 14:15:00 | 607.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 10:45:00 | 639.00 | 2025-09-26 14:15:00 | 607.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 12:00:00 | 639.40 | 2025-09-26 14:15:00 | 607.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 10:15:00 | 639.00 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2025-09-12 10:45:00 | 639.00 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2025-09-12 12:00:00 | 639.40 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.13% |
