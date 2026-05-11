# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 2748.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 148 |
| ALERT1 | 102 |
| ALERT2 | 101 |
| ALERT2_SKIP | 54 |
| ALERT3 | 281 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 138 |
| PARTIAL | 11 |
| TARGET_HIT | 20 |
| STOP_HIT | 117 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 148 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 58 / 90
- **Target hits / Stop hits / Partials:** 20 / 117 / 11
- **Avg / median % per leg:** 0.58% / -0.98%
- **Sum % (uncompounded):** 85.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 29 | 39.7% | 17 | 56 | 0 | 1.40% | 102.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 73 | 29 | 39.7% | 17 | 56 | 0 | 1.40% | 102.4% |
| SELL (all) | 75 | 29 | 38.7% | 3 | 61 | 11 | -0.22% | -16.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.47% | -0.5% |
| SELL @ 3rd Alert (retest2) | 74 | 29 | 39.2% | 3 | 60 | 11 | -0.22% | -16.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.47% | -0.5% |
| retest2 (combined) | 147 | 58 | 39.5% | 20 | 116 | 11 | 0.59% | 86.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 15:15:00 | 605.25 | 610.29 | 610.31 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 10:15:00 | 611.35 | 610.44 | 610.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 11:15:00 | 616.80 | 611.71 | 610.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 14:15:00 | 614.98 | 615.22 | 613.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 15:00:00 | 614.98 | 615.22 | 613.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 618.50 | 615.88 | 613.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:45:00 | 619.33 | 616.50 | 614.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:00:00 | 620.50 | 621.72 | 619.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 15:15:00 | 614.00 | 617.97 | 618.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 614.00 | 617.97 | 618.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 611.38 | 616.65 | 617.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 611.75 | 607.39 | 610.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 13:15:00 | 611.75 | 607.39 | 610.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 611.75 | 607.39 | 610.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 611.75 | 607.39 | 610.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 606.65 | 607.24 | 610.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 09:30:00 | 603.50 | 606.38 | 609.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 624.90 | 605.23 | 606.34 | SL hit (close>static) qty=1.00 sl=612.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 10:15:00 | 618.45 | 607.88 | 607.44 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 601.10 | 607.25 | 607.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 594.55 | 603.90 | 605.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 12:15:00 | 607.70 | 603.81 | 605.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 607.70 | 603.81 | 605.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 607.70 | 603.81 | 605.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 607.35 | 603.81 | 605.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 609.28 | 604.90 | 605.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:15:00 | 610.50 | 604.90 | 605.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 15:15:00 | 610.95 | 606.92 | 606.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 09:15:00 | 617.40 | 609.02 | 607.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 616.23 | 620.45 | 615.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 616.23 | 620.45 | 615.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 616.23 | 620.45 | 615.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 616.23 | 620.45 | 615.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 615.80 | 619.52 | 615.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:15:00 | 615.40 | 619.52 | 615.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 615.23 | 618.66 | 615.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:45:00 | 614.88 | 618.66 | 615.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 611.50 | 617.23 | 615.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 611.50 | 617.23 | 615.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 615.03 | 616.79 | 615.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 15:00:00 | 619.88 | 617.41 | 615.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 607.83 | 613.80 | 614.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 607.83 | 613.80 | 614.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 11:15:00 | 604.48 | 611.93 | 613.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 561.55 | 559.82 | 573.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 561.55 | 559.82 | 573.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 576.40 | 561.52 | 569.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 576.40 | 561.52 | 569.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 583.67 | 565.95 | 570.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 583.67 | 565.95 | 570.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 605.73 | 578.58 | 575.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 608.00 | 594.76 | 585.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 14:15:00 | 648.28 | 659.67 | 651.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 14:15:00 | 648.28 | 659.67 | 651.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 648.28 | 659.67 | 651.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 648.28 | 659.67 | 651.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 649.50 | 657.64 | 651.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 645.00 | 657.64 | 651.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 645.73 | 655.25 | 651.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:45:00 | 657.00 | 654.95 | 651.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 10:15:00 | 722.70 | 674.47 | 662.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 659.05 | 678.22 | 679.91 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 15:15:00 | 671.75 | 667.97 | 667.83 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 661.50 | 666.58 | 667.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 652.35 | 663.73 | 665.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 663.13 | 662.63 | 664.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 663.13 | 662.63 | 664.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 663.13 | 662.63 | 664.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 661.60 | 662.63 | 664.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 661.35 | 662.37 | 664.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:45:00 | 662.53 | 662.37 | 664.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 649.88 | 652.45 | 657.72 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 664.05 | 659.34 | 659.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 690.00 | 666.82 | 662.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 688.48 | 688.51 | 678.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 11:00:00 | 688.48 | 688.51 | 678.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 702.95 | 694.55 | 686.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:45:00 | 720.53 | 701.72 | 691.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 11:15:00 | 689.40 | 698.92 | 699.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 11:15:00 | 689.40 | 698.92 | 699.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 668.23 | 687.09 | 692.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 12:15:00 | 682.85 | 682.23 | 688.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 12:45:00 | 683.63 | 682.23 | 688.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 684.65 | 681.55 | 686.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 680.03 | 681.55 | 686.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 679.30 | 681.10 | 686.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:45:00 | 675.50 | 679.95 | 685.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:00:00 | 675.00 | 678.96 | 684.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 13:00:00 | 675.50 | 678.27 | 683.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:45:00 | 676.35 | 678.38 | 682.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 680.40 | 678.40 | 681.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 680.10 | 678.40 | 681.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 698.58 | 682.44 | 683.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 698.58 | 682.44 | 683.32 | SL hit (close>static) qty=1.00 sl=689.18 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 697.20 | 685.39 | 684.58 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 11:15:00 | 680.03 | 684.13 | 684.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 679.50 | 683.20 | 683.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 14:15:00 | 685.15 | 683.40 | 683.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 14:15:00 | 685.15 | 683.40 | 683.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 685.15 | 683.40 | 683.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 685.15 | 683.40 | 683.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 681.00 | 682.92 | 683.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 685.65 | 682.92 | 683.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 681.90 | 682.71 | 683.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:15:00 | 680.08 | 682.67 | 683.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:00:00 | 680.53 | 682.24 | 683.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 646.08 | 660.39 | 669.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 646.50 | 660.39 | 669.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-19 11:15:00 | 663.78 | 658.81 | 666.74 | SL hit (close>ema200) qty=0.50 sl=658.81 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 674.30 | 654.28 | 652.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 14:15:00 | 678.95 | 671.21 | 665.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 670.08 | 672.69 | 666.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 670.08 | 672.69 | 666.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 669.00 | 669.98 | 667.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 669.93 | 669.98 | 667.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 666.13 | 669.21 | 667.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 667.90 | 669.21 | 667.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 663.68 | 668.10 | 667.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 663.68 | 668.10 | 667.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 665.00 | 667.48 | 667.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:15:00 | 667.50 | 667.48 | 667.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:30:00 | 667.50 | 672.19 | 671.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 15:15:00 | 669.45 | 672.19 | 671.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 650.65 | 667.44 | 669.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 650.65 | 667.44 | 669.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 14:15:00 | 609.30 | 618.81 | 627.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 620.45 | 616.14 | 621.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 14:00:00 | 620.45 | 616.14 | 621.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 629.23 | 618.76 | 622.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 629.23 | 618.76 | 622.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 631.00 | 621.21 | 623.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 640.50 | 621.21 | 623.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 639.92 | 624.95 | 624.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 646.30 | 637.36 | 632.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 653.70 | 658.38 | 653.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 653.70 | 658.38 | 653.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 653.70 | 658.38 | 653.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 653.70 | 658.38 | 653.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 652.83 | 657.27 | 653.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:45:00 | 651.78 | 657.27 | 653.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 648.98 | 655.61 | 652.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 648.98 | 655.61 | 652.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 648.25 | 654.14 | 652.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:30:00 | 648.55 | 654.14 | 652.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 647.53 | 652.44 | 652.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 647.53 | 652.44 | 652.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 647.75 | 651.50 | 651.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 09:15:00 | 640.17 | 649.24 | 650.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 11:15:00 | 650.08 | 648.09 | 649.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 11:15:00 | 650.08 | 648.09 | 649.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 650.08 | 648.09 | 649.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:30:00 | 650.00 | 648.09 | 649.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 652.50 | 648.97 | 649.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 652.50 | 648.97 | 649.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 652.88 | 649.76 | 650.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 653.20 | 649.76 | 650.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 655.17 | 650.84 | 650.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 664.08 | 654.15 | 652.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 11:15:00 | 650.00 | 654.59 | 652.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 11:15:00 | 650.00 | 654.59 | 652.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 650.00 | 654.59 | 652.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 650.00 | 654.59 | 652.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 648.05 | 653.28 | 652.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:30:00 | 649.00 | 653.28 | 652.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 645.90 | 651.81 | 651.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 643.48 | 650.14 | 651.09 | Break + close below crossover candle low |

### Cycle 22 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 673.25 | 654.10 | 652.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 13:15:00 | 684.75 | 669.51 | 661.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 676.50 | 676.50 | 667.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 10:45:00 | 678.45 | 676.50 | 667.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 682.05 | 687.05 | 682.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 685.23 | 687.05 | 682.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 679.20 | 685.48 | 681.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 679.20 | 685.48 | 681.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 676.95 | 683.77 | 681.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:45:00 | 676.70 | 683.77 | 681.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 685.00 | 683.41 | 681.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:00:00 | 690.00 | 684.33 | 682.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:15:00 | 690.35 | 686.06 | 684.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 12:15:00 | 707.50 | 717.86 | 718.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 707.50 | 717.86 | 718.26 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 726.50 | 719.59 | 719.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 732.00 | 722.07 | 720.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 763.85 | 765.52 | 755.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 763.85 | 765.52 | 755.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 763.85 | 765.52 | 755.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:45:00 | 757.85 | 765.52 | 755.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 771.30 | 764.98 | 759.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:30:00 | 774.55 | 766.83 | 760.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 12:00:00 | 771.40 | 780.55 | 773.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 12:30:00 | 773.63 | 777.84 | 772.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 771.33 | 773.95 | 771.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 799.55 | 779.07 | 774.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 12:15:00 | 816.83 | 786.96 | 778.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-23 09:15:00 | 848.54 | 811.40 | 795.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 844.93 | 852.98 | 853.57 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 14:15:00 | 869.08 | 856.58 | 855.12 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 839.63 | 855.97 | 857.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 824.18 | 841.45 | 849.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 836.15 | 835.16 | 843.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:00:00 | 836.15 | 835.16 | 843.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 838.50 | 836.32 | 840.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 853.90 | 836.32 | 840.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 848.98 | 838.85 | 841.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 840.68 | 838.85 | 841.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 798.65 | 835.08 | 839.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 827.78 | 817.29 | 826.17 | SL hit (close>ema200) qty=0.50 sl=817.29 alert=retest2 |

### Cycle 28 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 842.25 | 831.07 | 829.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 848.50 | 837.66 | 833.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 836.78 | 839.23 | 835.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 836.78 | 839.23 | 835.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 844.50 | 840.28 | 835.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:45:00 | 853.30 | 844.49 | 838.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 10:15:00 | 852.53 | 850.46 | 845.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 14:15:00 | 831.83 | 841.77 | 842.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 831.83 | 841.77 | 842.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 829.80 | 836.62 | 839.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 11:15:00 | 832.90 | 831.08 | 834.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 11:15:00 | 832.90 | 831.08 | 834.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 832.90 | 831.08 | 834.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:30:00 | 834.00 | 831.08 | 834.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 780.65 | 789.36 | 799.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 774.38 | 785.29 | 795.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 775.38 | 783.03 | 793.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:30:00 | 772.08 | 781.93 | 790.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 735.66 | 759.36 | 774.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 736.61 | 759.36 | 774.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 733.48 | 759.36 | 774.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 761.03 | 759.69 | 772.82 | SL hit (close>ema200) qty=0.50 sl=759.69 alert=retest2 |

### Cycle 30 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 789.60 | 770.94 | 769.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 12:15:00 | 798.88 | 776.53 | 771.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 977.50 | 977.59 | 944.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 977.50 | 977.59 | 944.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1032.25 | 1040.13 | 1026.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:45:00 | 1030.00 | 1040.13 | 1026.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1034.50 | 1039.00 | 1027.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1044.93 | 1039.00 | 1027.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1056.47 | 1042.49 | 1030.27 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1015.00 | 1026.06 | 1026.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 12:15:00 | 990.00 | 1011.37 | 1019.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1020.00 | 999.85 | 1009.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1020.00 | 999.85 | 1009.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1020.00 | 999.85 | 1009.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 1023.65 | 999.85 | 1009.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1040.00 | 1007.88 | 1012.61 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 12:15:00 | 1032.22 | 1017.64 | 1016.53 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 990.00 | 1014.59 | 1015.79 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 1065.00 | 1010.35 | 1010.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1076.15 | 1039.05 | 1028.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 12:15:00 | 1019.68 | 1047.12 | 1035.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 12:15:00 | 1019.68 | 1047.12 | 1035.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 1019.68 | 1047.12 | 1035.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:45:00 | 1032.55 | 1047.12 | 1035.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 1019.75 | 1041.65 | 1034.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:30:00 | 1009.28 | 1041.65 | 1034.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1058.10 | 1036.47 | 1032.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 1060.72 | 1045.39 | 1039.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 11:45:00 | 1070.00 | 1053.03 | 1044.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 1060.35 | 1048.51 | 1045.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:15:00 | 1068.00 | 1053.54 | 1049.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 1069.95 | 1065.81 | 1058.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:45:00 | 1057.50 | 1065.81 | 1058.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 1065.55 | 1068.42 | 1063.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:30:00 | 1067.47 | 1068.42 | 1063.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 1061.28 | 1066.70 | 1063.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:00:00 | 1061.28 | 1066.70 | 1063.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 1062.00 | 1065.76 | 1063.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:15:00 | 1061.00 | 1065.76 | 1063.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 1061.00 | 1064.81 | 1062.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 1073.70 | 1064.81 | 1062.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 1054.47 | 1063.47 | 1062.70 | SL hit (close<static) qty=1.00 sl=1057.50 alert=retest2 |

### Cycle 35 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 1054.47 | 1061.67 | 1061.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 1048.70 | 1057.36 | 1059.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 1057.33 | 1055.13 | 1057.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 1057.33 | 1055.13 | 1057.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1057.33 | 1055.13 | 1057.97 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 1072.50 | 1062.04 | 1060.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 1142.97 | 1082.60 | 1071.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 15:15:00 | 1127.50 | 1128.90 | 1114.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 09:15:00 | 1143.45 | 1128.90 | 1114.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1136.60 | 1130.44 | 1116.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 10:45:00 | 1148.38 | 1133.65 | 1119.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 12:15:00 | 1147.90 | 1135.07 | 1121.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 12:45:00 | 1146.30 | 1136.80 | 1123.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 1108.18 | 1122.60 | 1123.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 1108.18 | 1122.60 | 1123.47 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 1136.38 | 1123.37 | 1122.01 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 14:15:00 | 1100.00 | 1119.34 | 1121.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 13:15:00 | 1083.28 | 1100.77 | 1108.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 1078.40 | 1073.08 | 1084.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1078.40 | 1073.08 | 1084.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1078.40 | 1073.08 | 1084.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 1084.93 | 1073.08 | 1084.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 1029.78 | 1025.69 | 1040.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:00:00 | 1029.78 | 1025.69 | 1040.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 1038.45 | 1028.24 | 1040.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:45:00 | 1038.13 | 1028.24 | 1040.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 1042.55 | 1031.10 | 1040.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 1033.93 | 1031.10 | 1040.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 1047.85 | 1045.25 | 1044.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 1047.85 | 1045.25 | 1044.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 1075.38 | 1051.25 | 1047.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 1060.03 | 1063.46 | 1055.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 1060.03 | 1063.46 | 1055.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1055.50 | 1061.87 | 1055.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1055.50 | 1061.87 | 1055.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1065.00 | 1062.50 | 1056.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 1050.47 | 1062.50 | 1056.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1044.45 | 1058.89 | 1055.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 1041.88 | 1058.89 | 1055.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1052.10 | 1057.53 | 1055.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1059.93 | 1057.53 | 1055.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 13:15:00 | 1059.08 | 1055.46 | 1054.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-30 09:15:00 | 1165.92 | 1124.33 | 1119.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 1082.00 | 1110.24 | 1113.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 1071.40 | 1098.58 | 1107.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 1083.93 | 1078.94 | 1089.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 1083.93 | 1078.94 | 1089.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 1052.75 | 1053.99 | 1063.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 1036.72 | 1053.19 | 1062.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 15:00:00 | 1048.90 | 1052.33 | 1061.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 1064.88 | 1057.74 | 1060.56 | SL hit (close>static) qty=1.00 sl=1064.50 alert=retest2 |

### Cycle 42 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 1085.75 | 1061.61 | 1058.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 1103.70 | 1074.44 | 1065.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 10:15:00 | 1136.45 | 1139.15 | 1115.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 11:00:00 | 1136.45 | 1139.15 | 1115.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 1113.55 | 1135.34 | 1117.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:45:00 | 1102.33 | 1135.34 | 1117.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 1102.03 | 1128.68 | 1116.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 1102.03 | 1128.68 | 1116.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 1096.70 | 1122.28 | 1114.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:30:00 | 1097.55 | 1122.28 | 1114.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 1051.90 | 1103.36 | 1106.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 1037.50 | 1069.96 | 1087.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 1008.55 | 1008.03 | 1036.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 1008.55 | 1008.03 | 1036.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1003.50 | 1011.37 | 1026.08 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 1038.72 | 1027.89 | 1027.54 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 09:15:00 | 1019.73 | 1026.58 | 1027.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 11:15:00 | 1007.03 | 1020.96 | 1024.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 1015.93 | 1004.15 | 1011.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 12:15:00 | 1015.93 | 1004.15 | 1011.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 1015.93 | 1004.15 | 1011.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 1015.93 | 1004.15 | 1011.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 1013.48 | 1006.01 | 1011.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:45:00 | 1003.75 | 1010.24 | 1012.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 1003.63 | 1010.24 | 1012.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:45:00 | 996.10 | 1008.09 | 1011.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 1000.68 | 1006.61 | 1010.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 984.03 | 997.61 | 1003.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 11:30:00 | 977.85 | 989.93 | 999.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 13:30:00 | 977.73 | 986.02 | 995.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 1010.28 | 990.46 | 995.26 | SL hit (close>static) qty=1.00 sl=1009.75 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 1005.90 | 999.01 | 998.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 1013.03 | 1001.81 | 999.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 994.10 | 1000.48 | 999.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 994.10 | 1000.48 | 999.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 994.10 | 1000.48 | 999.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 994.10 | 1000.48 | 999.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1000.98 | 1000.58 | 999.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:30:00 | 999.50 | 1000.58 | 999.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1000.80 | 1000.62 | 999.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:30:00 | 1001.98 | 1000.62 | 999.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 999.45 | 1000.39 | 999.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:30:00 | 998.23 | 1000.39 | 999.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 996.25 | 999.56 | 999.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:30:00 | 996.65 | 999.56 | 999.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 991.13 | 997.88 | 998.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 973.45 | 992.13 | 995.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 955.00 | 949.74 | 963.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:45:00 | 955.08 | 949.74 | 963.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 957.13 | 951.21 | 963.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 950.28 | 948.96 | 961.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 1018.33 | 961.32 | 964.61 | SL hit (close>static) qty=1.00 sl=975.50 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 1080.65 | 985.19 | 975.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 1090.50 | 1006.25 | 985.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 1288.53 | 1299.30 | 1275.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 1288.53 | 1299.30 | 1275.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1288.53 | 1299.30 | 1275.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:15:00 | 1279.13 | 1299.30 | 1275.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1290.30 | 1297.50 | 1276.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 11:30:00 | 1293.18 | 1297.44 | 1278.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 12:30:00 | 1293.97 | 1296.35 | 1279.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 1252.85 | 1273.54 | 1274.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1252.85 | 1273.54 | 1274.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 14:15:00 | 1246.15 | 1268.06 | 1271.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 1240.00 | 1230.81 | 1246.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 1240.00 | 1230.81 | 1246.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 1240.00 | 1230.81 | 1246.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 1240.00 | 1230.81 | 1246.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 1244.50 | 1233.55 | 1246.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 1210.10 | 1233.55 | 1246.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 12:15:00 | 1149.59 | 1193.83 | 1221.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 10:15:00 | 1178.75 | 1171.63 | 1197.90 | SL hit (close>ema200) qty=0.50 sl=1171.63 alert=retest2 |

### Cycle 50 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 1122.08 | 1109.14 | 1107.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 1143.05 | 1115.93 | 1110.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1114.00 | 1120.88 | 1114.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1114.00 | 1120.88 | 1114.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1114.00 | 1120.88 | 1114.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1114.00 | 1120.88 | 1114.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1098.35 | 1116.37 | 1112.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:45:00 | 1108.85 | 1116.37 | 1112.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1131.50 | 1119.40 | 1114.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:00:00 | 1138.38 | 1123.74 | 1118.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 15:15:00 | 1134.50 | 1135.25 | 1127.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 10:30:00 | 1146.95 | 1138.30 | 1130.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 1102.50 | 1128.94 | 1132.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 1102.50 | 1128.94 | 1132.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 1096.78 | 1122.51 | 1128.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 1073.38 | 1069.66 | 1086.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:45:00 | 1077.43 | 1069.66 | 1086.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1105.75 | 1078.00 | 1087.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1105.75 | 1078.00 | 1087.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1116.47 | 1085.69 | 1090.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:45:00 | 1120.00 | 1085.69 | 1090.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 1125.47 | 1093.65 | 1093.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 1128.25 | 1104.86 | 1098.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1219.97 | 1224.56 | 1208.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 10:00:00 | 1219.97 | 1224.56 | 1208.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1193.15 | 1216.61 | 1207.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 1193.15 | 1216.61 | 1207.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1186.83 | 1210.65 | 1205.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:45:00 | 1187.53 | 1210.65 | 1205.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 1156.90 | 1193.46 | 1198.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1144.88 | 1177.75 | 1189.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1102.05 | 1098.92 | 1113.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 09:30:00 | 1106.08 | 1098.92 | 1113.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 1125.05 | 1105.60 | 1114.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 1125.05 | 1105.60 | 1114.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 1101.08 | 1104.69 | 1113.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 1098.00 | 1102.93 | 1110.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 09:30:00 | 1092.65 | 1099.94 | 1108.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 12:15:00 | 1134.75 | 1112.39 | 1112.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 1134.75 | 1112.39 | 1112.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1175.10 | 1130.70 | 1121.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1151.35 | 1162.23 | 1146.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 1151.35 | 1162.23 | 1146.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 1212.50 | 1195.16 | 1179.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 1212.50 | 1195.16 | 1179.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1217.58 | 1227.02 | 1212.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1213.00 | 1227.02 | 1212.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1211.85 | 1223.98 | 1212.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 1212.35 | 1223.98 | 1212.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1221.00 | 1223.39 | 1213.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 1223.03 | 1223.39 | 1213.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 1231.03 | 1220.00 | 1214.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 13:15:00 | 1206.72 | 1215.58 | 1213.65 | SL hit (close<static) qty=1.00 sl=1207.85 alert=retest2 |

### Cycle 55 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 1190.58 | 1210.58 | 1211.55 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1257.00 | 1213.49 | 1211.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 1267.35 | 1240.44 | 1228.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 1209.38 | 1243.44 | 1234.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 1209.38 | 1243.44 | 1234.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1209.38 | 1243.44 | 1234.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:45:00 | 1201.72 | 1243.44 | 1234.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1209.50 | 1236.65 | 1232.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 11:30:00 | 1232.90 | 1235.47 | 1232.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 1191.93 | 1239.62 | 1239.02 | SL hit (close<static) qty=1.00 sl=1202.78 alert=retest2 |

### Cycle 57 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1202.70 | 1232.23 | 1235.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1158.47 | 1210.51 | 1224.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 1103.22 | 1087.30 | 1122.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:00:00 | 1103.22 | 1087.30 | 1122.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 1080.20 | 1097.32 | 1115.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 1065.50 | 1090.96 | 1111.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:30:00 | 1066.93 | 1080.75 | 1101.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 1126.60 | 1093.52 | 1097.70 | SL hit (close>static) qty=1.00 sl=1115.97 alert=retest2 |

### Cycle 58 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 1129.03 | 1103.05 | 1101.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 1135.00 | 1109.44 | 1104.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 1164.00 | 1165.21 | 1149.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:30:00 | 1160.35 | 1165.21 | 1149.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 1161.30 | 1168.29 | 1160.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 1139.50 | 1168.29 | 1160.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1134.55 | 1161.54 | 1158.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 10:00:00 | 1134.55 | 1161.54 | 1158.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 1137.75 | 1156.78 | 1156.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:15:00 | 1134.35 | 1156.78 | 1156.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 11:15:00 | 1138.40 | 1153.11 | 1154.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 14:15:00 | 1132.90 | 1144.96 | 1150.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 13:15:00 | 1101.45 | 1101.28 | 1114.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-23 14:00:00 | 1101.45 | 1101.28 | 1114.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1111.00 | 1103.58 | 1111.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:15:00 | 1126.20 | 1103.58 | 1111.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1121.30 | 1107.12 | 1112.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 15:00:00 | 1103.80 | 1110.56 | 1113.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:15:00 | 1048.61 | 1090.88 | 1102.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-25 15:15:00 | 1077.40 | 1070.82 | 1086.34 | SL hit (close>ema200) qty=0.50 sl=1070.82 alert=retest2 |

### Cycle 60 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1102.10 | 1087.82 | 1086.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 10:15:00 | 1124.10 | 1095.08 | 1090.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 1094.20 | 1096.95 | 1091.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 13:00:00 | 1094.20 | 1096.95 | 1091.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 1101.70 | 1097.90 | 1092.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:30:00 | 1100.00 | 1097.90 | 1092.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 1085.00 | 1095.32 | 1092.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 1085.00 | 1095.32 | 1092.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 1097.00 | 1095.66 | 1092.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 1090.90 | 1095.66 | 1092.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1116.40 | 1099.81 | 1094.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 10:15:00 | 1124.50 | 1099.81 | 1094.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 14:45:00 | 1126.50 | 1115.21 | 1105.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-02 13:15:00 | 1236.95 | 1157.03 | 1132.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 1125.00 | 1152.31 | 1154.76 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 1178.60 | 1155.46 | 1154.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 10:15:00 | 1195.70 | 1163.50 | 1158.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 1166.00 | 1173.23 | 1165.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 1166.00 | 1173.23 | 1165.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1166.00 | 1173.23 | 1165.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1166.00 | 1173.23 | 1165.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1151.50 | 1168.88 | 1164.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1132.30 | 1168.88 | 1164.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1144.70 | 1164.05 | 1162.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:30:00 | 1142.40 | 1164.05 | 1162.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 1140.70 | 1159.38 | 1160.58 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1219.60 | 1169.39 | 1164.21 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 10:15:00 | 1173.40 | 1178.01 | 1178.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 12:15:00 | 1155.80 | 1172.80 | 1175.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 1151.20 | 1133.51 | 1142.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 1151.20 | 1133.51 | 1142.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1151.20 | 1133.51 | 1142.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:15:00 | 1179.70 | 1133.51 | 1142.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 1188.60 | 1144.53 | 1146.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 1188.60 | 1144.53 | 1146.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 1178.00 | 1151.22 | 1149.45 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 13:15:00 | 1160.40 | 1180.58 | 1180.66 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 1197.00 | 1183.05 | 1181.44 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 1170.10 | 1182.95 | 1183.51 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 1184.10 | 1181.73 | 1181.71 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 1181.00 | 1181.59 | 1181.65 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 1185.50 | 1182.28 | 1181.94 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 1179.90 | 1181.42 | 1181.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1175.20 | 1179.15 | 1180.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 15:15:00 | 1178.00 | 1177.32 | 1178.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 15:15:00 | 1178.00 | 1177.32 | 1178.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1178.00 | 1177.32 | 1178.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 1166.80 | 1177.32 | 1178.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1162.40 | 1174.34 | 1177.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:15:00 | 1157.00 | 1174.34 | 1177.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:00:00 | 1160.70 | 1171.61 | 1175.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 1155.80 | 1137.42 | 1134.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 1155.80 | 1137.42 | 1134.96 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 1118.60 | 1136.00 | 1136.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 1111.20 | 1122.56 | 1128.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 14:15:00 | 1126.90 | 1121.79 | 1125.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 1126.90 | 1121.79 | 1125.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 1126.90 | 1121.79 | 1125.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 1126.90 | 1121.79 | 1125.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1121.90 | 1121.81 | 1125.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 1127.10 | 1121.81 | 1125.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1125.50 | 1122.55 | 1125.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:45:00 | 1118.80 | 1122.14 | 1124.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 1116.10 | 1120.93 | 1123.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 1119.20 | 1113.93 | 1118.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:30:00 | 1118.40 | 1115.73 | 1118.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1106.60 | 1103.97 | 1109.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 1106.60 | 1103.97 | 1109.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1099.50 | 1103.08 | 1108.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:15:00 | 1109.80 | 1103.08 | 1108.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1109.80 | 1104.42 | 1109.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 1091.60 | 1104.42 | 1109.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1075.80 | 1098.70 | 1105.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-18 14:15:00 | 1108.50 | 1099.81 | 1099.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 1108.50 | 1099.81 | 1099.06 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1090.80 | 1097.30 | 1098.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1081.20 | 1094.08 | 1096.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 14:15:00 | 1101.00 | 1094.72 | 1096.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 14:15:00 | 1101.00 | 1094.72 | 1096.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1101.00 | 1094.72 | 1096.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 1101.00 | 1094.72 | 1096.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1081.20 | 1092.01 | 1095.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1102.70 | 1094.15 | 1095.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1102.40 | 1095.80 | 1096.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1103.00 | 1095.80 | 1096.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 1100.50 | 1097.27 | 1096.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1134.30 | 1104.81 | 1100.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1096.40 | 1110.04 | 1103.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1096.40 | 1110.04 | 1103.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1096.40 | 1110.04 | 1103.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 1099.00 | 1110.04 | 1103.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1096.90 | 1107.41 | 1103.34 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 1092.50 | 1101.10 | 1101.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 1091.30 | 1095.86 | 1098.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1094.70 | 1094.43 | 1096.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 1094.70 | 1094.43 | 1096.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1094.70 | 1094.43 | 1096.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:30:00 | 1098.30 | 1094.43 | 1096.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1099.00 | 1095.35 | 1097.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:45:00 | 1095.40 | 1095.35 | 1097.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 1094.40 | 1095.16 | 1096.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:30:00 | 1092.80 | 1094.95 | 1096.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 14:15:00 | 1092.70 | 1094.76 | 1096.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 1100.00 | 1087.40 | 1090.23 | SL hit (close>static) qty=1.00 sl=1099.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 1100.20 | 1093.10 | 1092.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1131.90 | 1100.86 | 1096.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 14:15:00 | 1141.90 | 1142.85 | 1127.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 1141.90 | 1142.85 | 1127.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1132.50 | 1142.32 | 1130.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 1126.00 | 1142.32 | 1130.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1123.70 | 1138.60 | 1129.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1123.70 | 1138.60 | 1129.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1133.10 | 1137.50 | 1129.89 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 1099.80 | 1122.32 | 1124.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 15:15:00 | 1095.00 | 1106.29 | 1114.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 12:15:00 | 1110.70 | 1104.19 | 1110.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 12:15:00 | 1110.70 | 1104.19 | 1110.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1110.70 | 1104.19 | 1110.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 1112.50 | 1104.19 | 1110.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1104.40 | 1104.24 | 1109.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:45:00 | 1099.00 | 1102.87 | 1108.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1097.40 | 1103.29 | 1108.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 1120.00 | 1107.08 | 1108.95 | SL hit (close>static) qty=1.00 sl=1112.80 alert=retest2 |

### Cycle 82 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 1119.30 | 1110.26 | 1110.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 1120.90 | 1112.39 | 1111.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1113.90 | 1124.35 | 1120.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1113.90 | 1124.35 | 1120.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1113.90 | 1124.35 | 1120.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1113.90 | 1124.35 | 1120.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1113.50 | 1122.18 | 1119.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:15:00 | 1112.30 | 1122.18 | 1119.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 1107.60 | 1116.72 | 1117.57 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 1126.10 | 1117.44 | 1116.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 15:15:00 | 1130.10 | 1121.91 | 1119.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 15:15:00 | 1130.30 | 1130.55 | 1125.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:15:00 | 1129.70 | 1130.55 | 1125.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1127.40 | 1129.92 | 1126.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 1127.40 | 1129.92 | 1126.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1132.00 | 1130.34 | 1126.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:45:00 | 1144.60 | 1132.40 | 1128.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 1147.20 | 1132.40 | 1128.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 1159.00 | 1138.01 | 1132.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 1188.70 | 1200.77 | 1202.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 1188.70 | 1200.77 | 1202.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1181.60 | 1194.53 | 1198.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1195.90 | 1187.57 | 1192.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1195.90 | 1187.57 | 1192.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1195.90 | 1187.57 | 1192.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1195.90 | 1187.57 | 1192.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1201.80 | 1190.41 | 1193.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 1200.70 | 1190.41 | 1193.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 1187.30 | 1190.32 | 1192.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 1185.30 | 1189.05 | 1191.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1186.10 | 1189.24 | 1191.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 1179.80 | 1187.45 | 1190.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1190.00 | 1161.63 | 1161.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1190.00 | 1161.63 | 1161.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 1197.20 | 1173.89 | 1167.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 13:15:00 | 1164.60 | 1175.17 | 1169.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 13:15:00 | 1164.60 | 1175.17 | 1169.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1164.60 | 1175.17 | 1169.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:00:00 | 1164.60 | 1175.17 | 1169.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 1222.40 | 1184.61 | 1173.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 1247.50 | 1190.09 | 1177.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-01 09:15:00 | 1372.25 | 1293.15 | 1247.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 1295.40 | 1310.46 | 1310.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 1286.20 | 1305.61 | 1308.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 1300.00 | 1295.95 | 1302.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 1288.50 | 1295.95 | 1302.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1291.20 | 1295.00 | 1301.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 1270.80 | 1288.64 | 1295.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 1271.10 | 1282.14 | 1290.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 1271.40 | 1282.14 | 1290.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 14:15:00 | 1273.20 | 1279.17 | 1287.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1293.00 | 1281.60 | 1286.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 1309.20 | 1281.60 | 1286.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1300.40 | 1285.36 | 1288.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 1324.60 | 1295.65 | 1292.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 1324.60 | 1295.65 | 1292.52 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 1290.50 | 1301.78 | 1303.20 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1341.50 | 1309.26 | 1306.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 1353.80 | 1318.16 | 1310.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 1375.10 | 1376.52 | 1354.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:00:00 | 1375.10 | 1376.52 | 1354.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1422.00 | 1426.18 | 1414.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1407.20 | 1426.18 | 1414.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1409.10 | 1422.76 | 1414.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 13:45:00 | 1422.60 | 1415.91 | 1412.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 11:00:00 | 1423.20 | 1414.68 | 1413.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 1401.00 | 1411.25 | 1411.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 1401.00 | 1411.25 | 1411.98 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1422.50 | 1411.67 | 1411.61 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 1410.00 | 1411.52 | 1411.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 1405.30 | 1410.27 | 1411.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 1414.70 | 1411.16 | 1411.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 15:15:00 | 1414.70 | 1411.16 | 1411.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1414.70 | 1411.16 | 1411.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 1402.70 | 1411.16 | 1411.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1412.70 | 1411.47 | 1411.46 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 10:15:00 | 1410.00 | 1411.17 | 1411.32 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1415.90 | 1412.12 | 1411.74 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 1402.00 | 1410.64 | 1411.23 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1430.10 | 1411.85 | 1410.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 1461.10 | 1427.54 | 1419.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 09:15:00 | 1462.30 | 1465.62 | 1447.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 10:00:00 | 1462.30 | 1465.62 | 1447.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1474.50 | 1493.83 | 1480.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1474.50 | 1493.83 | 1480.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1471.30 | 1489.33 | 1479.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 1462.10 | 1489.33 | 1479.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1472.90 | 1486.04 | 1479.15 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 1457.00 | 1475.36 | 1475.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 1446.20 | 1465.19 | 1469.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 1458.00 | 1451.26 | 1459.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 11:15:00 | 1458.00 | 1451.26 | 1459.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1458.00 | 1451.26 | 1459.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 1458.00 | 1451.26 | 1459.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1456.30 | 1452.36 | 1458.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 1456.30 | 1452.36 | 1458.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1456.40 | 1453.17 | 1458.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1464.60 | 1453.17 | 1458.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1455.00 | 1453.53 | 1457.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1473.50 | 1453.53 | 1457.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1467.50 | 1456.33 | 1458.69 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1461.90 | 1459.82 | 1459.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 1474.30 | 1462.72 | 1460.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 1467.20 | 1468.48 | 1465.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:30:00 | 1467.00 | 1468.48 | 1465.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1466.00 | 1467.58 | 1465.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1466.00 | 1467.58 | 1465.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1467.00 | 1467.47 | 1465.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 1465.60 | 1467.47 | 1465.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1461.10 | 1474.61 | 1471.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1461.10 | 1474.61 | 1471.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1475.00 | 1474.69 | 1471.72 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 1460.80 | 1468.39 | 1469.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 1437.50 | 1455.96 | 1462.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 1461.00 | 1456.41 | 1461.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 1461.00 | 1456.41 | 1461.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1461.00 | 1456.41 | 1461.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 1463.50 | 1456.41 | 1461.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1454.50 | 1456.03 | 1460.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 1453.50 | 1454.62 | 1459.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 1453.50 | 1455.46 | 1459.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 1449.10 | 1453.86 | 1456.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1380.83 | 1412.73 | 1429.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1380.83 | 1412.73 | 1429.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1376.64 | 1412.73 | 1429.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-30 14:15:00 | 1308.15 | 1335.13 | 1355.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 102 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1385.00 | 1359.81 | 1357.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1392.50 | 1370.86 | 1363.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 1377.40 | 1382.13 | 1374.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 1377.40 | 1382.13 | 1374.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1389.60 | 1383.62 | 1375.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 1378.70 | 1383.62 | 1375.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1399.40 | 1394.48 | 1384.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 11:00:00 | 1406.00 | 1396.79 | 1386.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1425.00 | 1400.78 | 1392.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:45:00 | 1407.00 | 1403.72 | 1395.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 1408.00 | 1400.77 | 1397.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1480.60 | 1487.21 | 1471.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:45:00 | 1488.40 | 1486.76 | 1472.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 1494.40 | 1498.39 | 1481.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-15 10:15:00 | 1546.60 | 1506.71 | 1486.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 1700.30 | 1713.95 | 1714.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 1689.50 | 1709.06 | 1712.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 1712.20 | 1706.32 | 1710.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1712.20 | 1706.32 | 1710.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1712.20 | 1706.32 | 1710.22 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1752.90 | 1715.38 | 1712.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1777.20 | 1735.78 | 1725.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 1819.30 | 1823.60 | 1797.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:15:00 | 1784.50 | 1823.60 | 1797.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1769.50 | 1812.78 | 1794.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1769.50 | 1812.78 | 1794.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1778.60 | 1805.94 | 1793.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 1763.60 | 1805.94 | 1793.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1759.40 | 1784.66 | 1785.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 10:15:00 | 1753.50 | 1770.16 | 1776.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 1805.00 | 1767.01 | 1770.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1805.00 | 1767.01 | 1770.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1805.00 | 1767.01 | 1770.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 1805.00 | 1767.01 | 1770.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1786.60 | 1770.93 | 1771.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 1805.80 | 1770.93 | 1771.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 1790.60 | 1774.86 | 1773.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 1796.10 | 1779.11 | 1775.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 14:15:00 | 1781.30 | 1782.38 | 1777.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 15:00:00 | 1781.30 | 1782.38 | 1777.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1750.00 | 1775.90 | 1775.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 1693.20 | 1775.90 | 1775.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1688.30 | 1758.38 | 1767.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 10:15:00 | 1677.30 | 1742.17 | 1759.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 11:15:00 | 1708.70 | 1708.25 | 1726.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 11:45:00 | 1712.00 | 1708.25 | 1726.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1704.00 | 1699.46 | 1709.21 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 1729.50 | 1715.63 | 1714.14 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 1707.20 | 1713.13 | 1713.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1690.00 | 1708.00 | 1710.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 1707.80 | 1707.56 | 1709.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 12:30:00 | 1707.60 | 1707.56 | 1709.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1711.10 | 1708.26 | 1710.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 1711.10 | 1708.26 | 1710.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1709.40 | 1708.49 | 1710.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 1709.40 | 1708.49 | 1710.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 1714.90 | 1709.77 | 1710.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 1705.10 | 1709.77 | 1710.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1720.10 | 1711.84 | 1711.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 1720.10 | 1711.84 | 1711.32 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 1701.10 | 1711.16 | 1711.19 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 1716.90 | 1712.12 | 1711.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 1726.10 | 1716.56 | 1713.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 10:15:00 | 1713.90 | 1716.03 | 1713.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 1713.90 | 1716.03 | 1713.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1713.90 | 1716.03 | 1713.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 1713.90 | 1716.03 | 1713.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 1717.60 | 1716.34 | 1714.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:45:00 | 1727.10 | 1718.21 | 1715.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 1706.90 | 1720.44 | 1718.39 | SL hit (close<static) qty=1.00 sl=1710.10 alert=retest2 |

### Cycle 113 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 1697.50 | 1715.42 | 1716.66 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 1756.00 | 1721.62 | 1718.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 11:15:00 | 1837.80 | 1744.86 | 1729.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 1805.00 | 1807.77 | 1770.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:45:00 | 1826.70 | 1807.77 | 1770.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1771.00 | 1787.49 | 1777.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 1773.00 | 1787.49 | 1777.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1786.60 | 1787.31 | 1778.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 1772.40 | 1787.31 | 1778.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1793.30 | 1788.38 | 1781.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:30:00 | 1772.60 | 1788.38 | 1781.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1783.00 | 1787.24 | 1782.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:15:00 | 1780.80 | 1787.24 | 1782.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1767.30 | 1783.25 | 1780.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1767.30 | 1783.25 | 1780.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1788.10 | 1784.22 | 1781.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:30:00 | 1792.60 | 1786.52 | 1782.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 1792.10 | 1786.75 | 1783.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:30:00 | 1792.10 | 1793.85 | 1787.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1769.00 | 1801.45 | 1803.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 1769.00 | 1801.45 | 1803.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 1754.90 | 1786.69 | 1795.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 1711.00 | 1707.66 | 1722.12 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1693.50 | 1707.66 | 1722.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1683.90 | 1665.78 | 1687.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 1683.90 | 1665.78 | 1687.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1686.20 | 1669.86 | 1686.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:45:00 | 1687.90 | 1669.86 | 1686.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1686.00 | 1673.09 | 1686.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:00:00 | 1675.40 | 1673.55 | 1685.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1701.50 | 1676.50 | 1683.87 | SL hit (close>ema400) qty=1.00 sl=1683.87 alert=retest1 |

### Cycle 116 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1696.50 | 1689.13 | 1688.30 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1679.80 | 1686.28 | 1687.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 11:15:00 | 1666.50 | 1678.61 | 1683.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1667.30 | 1666.97 | 1674.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:45:00 | 1667.40 | 1666.97 | 1674.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1659.90 | 1665.56 | 1673.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 1654.00 | 1663.38 | 1671.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 1656.40 | 1662.75 | 1670.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1648.60 | 1663.12 | 1668.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1675.00 | 1653.78 | 1655.41 | SL hit (close>static) qty=1.00 sl=1673.90 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 1671.30 | 1657.28 | 1656.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 13:15:00 | 1688.30 | 1663.49 | 1659.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 1672.70 | 1674.83 | 1666.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 1672.70 | 1674.83 | 1666.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1668.20 | 1673.51 | 1666.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 1665.00 | 1673.51 | 1666.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1664.90 | 1671.79 | 1666.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 1664.90 | 1671.79 | 1666.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1662.00 | 1669.83 | 1666.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:30:00 | 1660.50 | 1669.83 | 1666.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1651.30 | 1666.12 | 1664.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 1651.30 | 1666.12 | 1664.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1660.10 | 1664.79 | 1664.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1640.20 | 1664.79 | 1664.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1630.10 | 1657.85 | 1661.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 12:15:00 | 1624.50 | 1633.06 | 1642.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 1633.90 | 1633.23 | 1641.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 1633.90 | 1633.23 | 1641.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1633.90 | 1633.23 | 1641.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 1633.10 | 1633.23 | 1641.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1647.20 | 1636.02 | 1642.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1648.00 | 1636.02 | 1642.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1648.00 | 1638.42 | 1642.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1668.00 | 1638.42 | 1642.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1674.20 | 1650.60 | 1647.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 1681.40 | 1660.13 | 1652.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1668.00 | 1669.37 | 1660.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1668.00 | 1669.37 | 1660.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1668.00 | 1669.37 | 1660.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1656.00 | 1669.37 | 1660.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1657.70 | 1667.04 | 1660.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:45:00 | 1658.60 | 1667.04 | 1660.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1649.00 | 1663.43 | 1659.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 1649.00 | 1663.43 | 1659.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1660.40 | 1671.86 | 1666.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1660.40 | 1671.86 | 1666.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1654.70 | 1668.42 | 1665.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 1654.70 | 1668.42 | 1665.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 1656.00 | 1663.23 | 1663.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1645.40 | 1657.86 | 1660.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 1661.20 | 1647.55 | 1653.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 1661.20 | 1647.55 | 1653.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1661.20 | 1647.55 | 1653.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 1661.20 | 1647.55 | 1653.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1650.00 | 1648.04 | 1653.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 1662.10 | 1648.04 | 1653.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1648.80 | 1648.19 | 1653.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 1695.40 | 1648.19 | 1653.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1680.00 | 1654.55 | 1655.58 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 10:15:00 | 1678.40 | 1659.32 | 1657.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 1694.60 | 1666.38 | 1661.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 13:15:00 | 1670.90 | 1671.40 | 1664.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:45:00 | 1675.50 | 1671.40 | 1664.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1651.40 | 1667.40 | 1663.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1651.40 | 1667.40 | 1663.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1660.00 | 1665.92 | 1662.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1671.30 | 1665.92 | 1662.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 1663.50 | 1665.44 | 1663.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 12:15:00 | 1652.00 | 1660.56 | 1661.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 1652.00 | 1660.56 | 1661.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 13:15:00 | 1644.50 | 1657.34 | 1659.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1660.00 | 1657.88 | 1659.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1660.00 | 1657.88 | 1659.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1660.00 | 1657.88 | 1659.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1660.00 | 1657.88 | 1659.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1658.00 | 1657.90 | 1659.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 1659.30 | 1657.32 | 1659.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1655.00 | 1656.86 | 1658.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 1661.60 | 1656.86 | 1658.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1657.80 | 1657.05 | 1658.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1657.80 | 1657.05 | 1658.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1668.00 | 1659.24 | 1659.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1668.00 | 1659.24 | 1659.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1661.00 | 1659.59 | 1659.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:15:00 | 1680.00 | 1659.59 | 1659.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1705.10 | 1668.69 | 1663.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 1733.00 | 1692.81 | 1677.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1749.30 | 1761.69 | 1747.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1749.30 | 1761.69 | 1747.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1749.30 | 1761.69 | 1747.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 1749.30 | 1761.69 | 1747.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1762.60 | 1761.87 | 1749.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 1770.90 | 1755.23 | 1748.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 1773.00 | 1757.53 | 1752.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 1770.00 | 1757.53 | 1752.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 1776.80 | 1767.76 | 1757.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1764.00 | 1767.82 | 1760.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1750.00 | 1767.82 | 1760.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1760.10 | 1766.28 | 1760.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 1750.30 | 1766.28 | 1760.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1748.00 | 1762.62 | 1759.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1748.00 | 1762.62 | 1759.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1754.00 | 1760.90 | 1758.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1754.00 | 1760.90 | 1758.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1754.40 | 1759.60 | 1758.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 1752.60 | 1759.60 | 1758.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 1739.00 | 1754.33 | 1756.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 1739.00 | 1754.33 | 1756.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 1732.00 | 1749.86 | 1753.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 1677.10 | 1676.62 | 1694.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 1677.10 | 1676.62 | 1694.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1693.10 | 1680.60 | 1690.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1741.70 | 1680.60 | 1690.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1732.90 | 1691.06 | 1694.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 1720.40 | 1691.06 | 1694.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1731.40 | 1699.13 | 1698.00 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1700.00 | 1705.43 | 1705.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 1681.90 | 1699.01 | 1702.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1651.50 | 1620.76 | 1637.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1651.50 | 1620.76 | 1637.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1651.50 | 1620.76 | 1637.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1651.50 | 1620.76 | 1637.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1630.80 | 1622.76 | 1637.24 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 1652.40 | 1642.25 | 1641.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 1695.70 | 1656.38 | 1648.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 1909.00 | 1921.37 | 1879.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 1909.00 | 1921.37 | 1879.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1909.00 | 1921.37 | 1879.18 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1824.00 | 1865.58 | 1868.58 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 09:15:00 | 1901.70 | 1872.80 | 1871.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 1938.00 | 1890.92 | 1881.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1969.00 | 1976.12 | 1954.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1969.00 | 1976.12 | 1954.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1969.00 | 1976.12 | 1954.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 1990.20 | 1978.95 | 1966.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 12:00:00 | 1990.30 | 1981.55 | 1969.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 12:30:00 | 1996.90 | 1984.24 | 1971.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1999.30 | 1986.03 | 1976.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1999.00 | 1988.63 | 1978.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 2023.40 | 1997.59 | 1987.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 2019.90 | 2057.82 | 2058.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 2019.90 | 2057.82 | 2058.60 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 2070.10 | 2054.53 | 2052.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 2081.80 | 2064.89 | 2058.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 2119.20 | 2123.31 | 2099.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 2103.50 | 2124.09 | 2109.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 2103.50 | 2124.09 | 2109.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 2103.50 | 2124.09 | 2109.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 2090.00 | 2117.27 | 2107.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 2086.20 | 2117.27 | 2107.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 2106.40 | 2112.37 | 2106.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:15:00 | 2112.30 | 2109.49 | 2106.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:45:00 | 2113.20 | 2110.26 | 2107.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 2142.70 | 2109.41 | 2107.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 2152.10 | 2163.43 | 2164.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 2152.10 | 2163.43 | 2164.59 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 09:15:00 | 2173.70 | 2165.48 | 2165.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 10:15:00 | 2186.00 | 2169.59 | 2167.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 12:15:00 | 2151.00 | 2166.82 | 2166.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 12:15:00 | 2151.00 | 2166.82 | 2166.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 2151.00 | 2166.82 | 2166.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:45:00 | 2146.50 | 2166.82 | 2166.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 2174.00 | 2168.25 | 2167.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:45:00 | 2188.40 | 2173.46 | 2169.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 2108.00 | 2164.46 | 2166.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2108.00 | 2164.46 | 2166.46 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 2220.00 | 2162.43 | 2156.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 2235.10 | 2176.97 | 2163.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2162.40 | 2228.63 | 2212.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2162.40 | 2228.63 | 2212.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2162.40 | 2228.63 | 2212.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:30:00 | 2133.70 | 2228.63 | 2212.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 2142.10 | 2211.33 | 2206.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 2149.00 | 2211.33 | 2206.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 2143.40 | 2197.74 | 2200.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 2133.70 | 2175.69 | 2189.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 2256.20 | 2187.00 | 2190.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 2256.20 | 2187.00 | 2190.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2256.20 | 2187.00 | 2190.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 2256.20 | 2187.00 | 2190.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 2245.30 | 2198.66 | 2195.56 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 2193.30 | 2204.84 | 2205.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2167.60 | 2194.22 | 2200.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 2179.50 | 2176.70 | 2188.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 2179.50 | 2176.70 | 2188.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 2179.50 | 2176.70 | 2188.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 2189.80 | 2176.70 | 2188.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 2186.80 | 2178.72 | 2188.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:30:00 | 2179.30 | 2178.72 | 2188.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 2180.00 | 2178.97 | 2187.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 2154.00 | 2178.97 | 2187.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 2305.80 | 2164.88 | 2151.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 2305.80 | 2164.88 | 2151.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 2316.00 | 2267.80 | 2254.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 2323.50 | 2338.51 | 2304.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 2323.50 | 2338.51 | 2304.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 2296.30 | 2330.07 | 2303.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 2311.80 | 2330.07 | 2303.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 2281.20 | 2320.30 | 2301.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:30:00 | 2281.50 | 2320.30 | 2301.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 2270.10 | 2301.87 | 2295.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:30:00 | 2271.60 | 2301.87 | 2295.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 2269.00 | 2290.81 | 2291.54 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 2450.80 | 2322.81 | 2306.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2543.50 | 2429.46 | 2374.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 12:15:00 | 2512.20 | 2516.66 | 2470.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 12:30:00 | 2511.00 | 2516.66 | 2470.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 2554.90 | 2532.35 | 2507.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 15:15:00 | 2572.00 | 2532.35 | 2507.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 2459.20 | 2524.06 | 2508.06 | SL hit (close<static) qty=1.00 sl=2496.10 alert=retest2 |

### Cycle 143 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 2390.40 | 2497.33 | 2497.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 11:15:00 | 2276.40 | 2453.14 | 2477.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 2166.60 | 2154.26 | 2245.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:45:00 | 2168.20 | 2154.26 | 2245.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 2245.00 | 2179.73 | 2241.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:00:00 | 2245.00 | 2179.73 | 2241.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 2210.90 | 2185.96 | 2238.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 15:15:00 | 2205.00 | 2185.96 | 2238.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2285.00 | 2231.70 | 2236.43 | SL hit (close>static) qty=1.00 sl=2247.70 alert=retest2 |

### Cycle 144 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 2287.80 | 2246.17 | 2242.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 2296.60 | 2256.26 | 2247.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 2321.20 | 2327.93 | 2295.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 2321.20 | 2327.93 | 2295.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2333.50 | 2354.51 | 2334.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 2348.00 | 2346.17 | 2333.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 2307.40 | 2337.17 | 2339.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 12:15:00 | 2307.40 | 2337.17 | 2339.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 13:15:00 | 2286.80 | 2327.10 | 2334.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 2325.40 | 2316.55 | 2325.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 10:15:00 | 2325.40 | 2316.55 | 2325.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 2325.40 | 2316.55 | 2325.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 2323.00 | 2316.55 | 2325.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 2309.00 | 2315.04 | 2324.28 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2374.00 | 2336.30 | 2331.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 2392.60 | 2369.84 | 2355.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 12:15:00 | 2397.60 | 2405.01 | 2385.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 12:15:00 | 2397.60 | 2405.01 | 2385.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 2397.60 | 2405.01 | 2385.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 2391.10 | 2405.01 | 2385.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 2388.00 | 2401.61 | 2385.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 2388.00 | 2401.61 | 2385.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 2376.40 | 2396.56 | 2384.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 2376.40 | 2396.56 | 2384.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 2362.50 | 2389.75 | 2382.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 2378.90 | 2389.75 | 2382.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2409.70 | 2418.42 | 2406.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 2421.50 | 2418.42 | 2406.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 2397.10 | 2414.16 | 2405.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 2399.60 | 2414.16 | 2405.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 2396.30 | 2410.59 | 2404.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:45:00 | 2396.60 | 2410.59 | 2404.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 2381.70 | 2398.79 | 2400.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 2371.90 | 2390.73 | 2396.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 12:15:00 | 2397.60 | 2384.65 | 2391.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 12:15:00 | 2397.60 | 2384.65 | 2391.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 2397.60 | 2384.65 | 2391.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 2408.00 | 2384.65 | 2391.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 2406.00 | 2388.92 | 2392.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 2406.00 | 2388.92 | 2392.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 2394.80 | 2391.71 | 2393.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 2424.00 | 2391.71 | 2393.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 2436.00 | 2400.57 | 2397.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 2454.00 | 2411.26 | 2402.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 2482.00 | 2487.53 | 2457.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:00:00 | 2482.00 | 2487.53 | 2457.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2470.00 | 2478.96 | 2460.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 2512.70 | 2478.96 | 2460.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 2490.50 | 2481.27 | 2463.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 2558.60 | 2495.52 | 2479.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 2575.40 | 2496.75 | 2484.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 11:15:00 | 617.65 | 2024-05-15 15:15:00 | 605.25 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-05-17 09:45:00 | 619.33 | 2024-05-21 15:15:00 | 614.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-05-21 11:00:00 | 620.50 | 2024-05-21 15:15:00 | 614.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-05-24 09:30:00 | 603.50 | 2024-05-27 09:15:00 | 624.90 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-05-30 15:00:00 | 619.88 | 2024-05-31 10:15:00 | 607.83 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-06-18 11:45:00 | 657.00 | 2024-06-19 10:15:00 | 722.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-05 11:45:00 | 720.53 | 2024-07-09 11:15:00 | 689.40 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2024-07-11 10:45:00 | 675.50 | 2024-07-12 10:15:00 | 698.58 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2024-07-11 12:00:00 | 675.00 | 2024-07-12 10:15:00 | 698.58 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-07-11 13:00:00 | 675.50 | 2024-07-12 10:15:00 | 698.58 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2024-07-11 14:45:00 | 676.35 | 2024-07-12 10:15:00 | 698.58 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-07-16 12:15:00 | 680.08 | 2024-07-19 09:15:00 | 646.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 13:00:00 | 680.53 | 2024-07-19 09:15:00 | 646.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 12:15:00 | 680.08 | 2024-07-19 11:15:00 | 663.78 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2024-07-16 13:00:00 | 680.53 | 2024-07-19 11:15:00 | 663.78 | STOP_HIT | 0.50 | 2.46% |
| BUY | retest2 | 2024-07-31 12:15:00 | 667.50 | 2024-08-05 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-08-02 14:30:00 | 667.50 | 2024-08-05 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-08-02 15:15:00 | 669.45 | 2024-08-05 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-09-03 12:00:00 | 690.00 | 2024-09-12 12:15:00 | 707.50 | STOP_HIT | 1.00 | 2.54% |
| BUY | retest2 | 2024-09-04 10:15:00 | 690.35 | 2024-09-12 12:15:00 | 707.50 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2024-09-18 10:30:00 | 774.55 | 2024-09-23 09:15:00 | 848.54 | TARGET_HIT | 1.00 | 9.55% |
| BUY | retest2 | 2024-09-19 12:00:00 | 771.40 | 2024-09-23 09:15:00 | 850.99 | TARGET_HIT | 1.00 | 10.32% |
| BUY | retest2 | 2024-09-19 12:30:00 | 773.63 | 2024-09-23 09:15:00 | 848.46 | TARGET_HIT | 1.00 | 9.67% |
| BUY | retest2 | 2024-09-20 09:15:00 | 771.33 | 2024-09-23 10:15:00 | 852.00 | TARGET_HIT | 1.00 | 10.46% |
| BUY | retest2 | 2024-09-20 12:15:00 | 816.83 | 2024-09-27 09:15:00 | 898.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-07 10:15:00 | 840.68 | 2024-10-07 10:15:00 | 798.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 10:15:00 | 840.68 | 2024-10-08 10:15:00 | 827.78 | STOP_HIT | 0.50 | 1.53% |
| BUY | retest2 | 2024-10-10 09:45:00 | 853.30 | 2024-10-11 14:15:00 | 831.83 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-10-11 10:15:00 | 852.53 | 2024-10-11 14:15:00 | 831.83 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-10-21 12:00:00 | 774.38 | 2024-10-22 15:15:00 | 735.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 775.38 | 2024-10-22 15:15:00 | 736.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:30:00 | 772.08 | 2024-10-22 15:15:00 | 733.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 774.38 | 2024-10-23 09:15:00 | 761.03 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2024-10-21 12:30:00 | 775.38 | 2024-10-23 09:15:00 | 761.03 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2024-10-22 09:30:00 | 772.08 | 2024-10-23 09:15:00 | 761.03 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2024-10-23 14:30:00 | 772.75 | 2024-10-28 10:15:00 | 785.60 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-10-24 12:15:00 | 769.85 | 2024-10-28 10:15:00 | 785.60 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-10-24 12:45:00 | 764.05 | 2024-10-28 10:15:00 | 785.60 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-10-24 14:30:00 | 769.10 | 2024-10-28 10:15:00 | 785.60 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-10-24 15:00:00 | 769.70 | 2024-10-28 10:15:00 | 785.60 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-10-25 15:15:00 | 750.00 | 2024-10-28 11:15:00 | 789.60 | STOP_HIT | 1.00 | -5.28% |
| BUY | retest2 | 2024-11-22 09:15:00 | 1060.72 | 2024-11-28 10:15:00 | 1054.47 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-11-22 11:45:00 | 1070.00 | 2024-11-28 11:15:00 | 1054.47 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-11-25 09:15:00 | 1060.35 | 2024-11-28 11:15:00 | 1054.47 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-11-25 14:15:00 | 1068.00 | 2024-11-28 11:15:00 | 1054.47 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-11-28 09:15:00 | 1073.70 | 2024-11-28 11:15:00 | 1054.47 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-04 10:45:00 | 1148.38 | 2024-12-06 09:15:00 | 1108.18 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-12-04 12:15:00 | 1147.90 | 2024-12-06 09:15:00 | 1108.18 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-12-04 12:45:00 | 1146.30 | 2024-12-06 09:15:00 | 1108.18 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1033.93 | 2024-12-19 14:15:00 | 1047.85 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-12-23 11:15:00 | 1059.93 | 2024-12-30 09:15:00 | 1165.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-23 13:15:00 | 1059.08 | 2024-12-30 09:15:00 | 1164.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-02 13:30:00 | 1036.72 | 2025-01-03 12:15:00 | 1064.88 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-01-02 15:00:00 | 1048.90 | 2025-01-03 12:15:00 | 1064.88 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-01-06 11:00:00 | 1049.15 | 2025-01-07 09:15:00 | 1084.65 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-01-06 13:00:00 | 1039.40 | 2025-01-07 09:15:00 | 1084.65 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2025-01-21 09:45:00 | 1003.75 | 2025-01-23 09:15:00 | 1010.28 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-01-21 10:15:00 | 1003.63 | 2025-01-23 09:15:00 | 1010.28 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-01-21 10:45:00 | 996.10 | 2025-01-23 13:15:00 | 1005.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-01-21 12:00:00 | 1000.68 | 2025-01-23 13:15:00 | 1005.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-01-22 11:30:00 | 977.85 | 2025-01-23 13:15:00 | 1005.90 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-01-22 13:30:00 | 977.73 | 2025-01-23 13:15:00 | 1005.90 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-01-28 14:45:00 | 950.28 | 2025-01-29 09:15:00 | 1018.33 | STOP_HIT | 1.00 | -7.16% |
| BUY | retest2 | 2025-02-06 11:30:00 | 1293.18 | 2025-02-07 13:15:00 | 1252.85 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-02-06 12:30:00 | 1293.97 | 2025-02-07 13:15:00 | 1252.85 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1210.10 | 2025-02-11 12:15:00 | 1149.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1210.10 | 2025-02-12 10:15:00 | 1178.75 | STOP_HIT | 0.50 | 2.59% |
| BUY | retest2 | 2025-02-24 10:00:00 | 1138.38 | 2025-02-27 11:15:00 | 1102.50 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-02-24 15:15:00 | 1134.50 | 2025-02-27 11:15:00 | 1102.50 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-02-25 10:30:00 | 1146.95 | 2025-02-27 11:15:00 | 1102.50 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-03-17 15:00:00 | 1098.00 | 2025-03-18 12:15:00 | 1134.75 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-03-18 09:30:00 | 1092.65 | 2025-03-18 12:15:00 | 1134.75 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2025-03-25 14:15:00 | 1223.03 | 2025-03-26 13:15:00 | 1206.72 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-26 10:45:00 | 1231.03 | 2025-03-26 13:15:00 | 1206.72 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-04-02 11:30:00 | 1232.90 | 2025-04-04 09:15:00 | 1191.93 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-04-09 11:00:00 | 1065.50 | 2025-04-11 12:15:00 | 1126.60 | STOP_HIT | 1.00 | -5.73% |
| SELL | retest2 | 2025-04-09 13:30:00 | 1066.93 | 2025-04-11 12:15:00 | 1126.60 | STOP_HIT | 1.00 | -5.59% |
| SELL | retest2 | 2025-04-24 15:00:00 | 1103.80 | 2025-04-25 10:15:00 | 1048.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 15:00:00 | 1103.80 | 2025-04-25 15:15:00 | 1077.40 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2025-04-29 09:15:00 | 1102.50 | 2025-04-29 09:15:00 | 1102.10 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-04-30 10:15:00 | 1124.50 | 2025-05-02 13:15:00 | 1236.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-30 14:45:00 | 1126.50 | 2025-05-02 13:15:00 | 1239.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-02 10:15:00 | 1157.00 | 2025-06-06 10:15:00 | 1155.80 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-06-02 11:00:00 | 1160.70 | 2025-06-06 10:15:00 | 1155.80 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-06-11 10:45:00 | 1118.80 | 2025-06-18 14:15:00 | 1108.50 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2025-06-11 12:00:00 | 1116.10 | 2025-06-18 14:15:00 | 1108.50 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1119.20 | 2025-06-18 14:15:00 | 1108.50 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-06-12 13:30:00 | 1118.40 | 2025-06-18 14:15:00 | 1108.50 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2025-06-25 12:30:00 | 1092.80 | 2025-06-26 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-25 14:15:00 | 1092.70 | 2025-06-26 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-03 14:45:00 | 1099.00 | 2025-07-04 11:15:00 | 1120.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-07-04 09:15:00 | 1097.40 | 2025-07-04 11:15:00 | 1120.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-11 13:45:00 | 1144.60 | 2025-07-21 13:15:00 | 1188.70 | STOP_HIT | 1.00 | 3.85% |
| BUY | retest2 | 2025-07-11 14:15:00 | 1147.20 | 2025-07-21 13:15:00 | 1188.70 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2025-07-14 10:15:00 | 1159.00 | 2025-07-21 13:15:00 | 1188.70 | STOP_HIT | 1.00 | 2.56% |
| SELL | retest2 | 2025-07-23 15:00:00 | 1185.30 | 2025-07-30 09:15:00 | 1190.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-24 09:15:00 | 1186.10 | 2025-07-30 09:15:00 | 1190.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1179.80 | 2025-07-30 09:15:00 | 1190.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-31 09:15:00 | 1247.50 | 2025-08-01 09:15:00 | 1372.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-11 09:15:00 | 1270.80 | 2025-08-12 11:15:00 | 1324.60 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2025-08-11 11:30:00 | 1271.10 | 2025-08-12 11:15:00 | 1324.60 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-08-11 12:15:00 | 1271.40 | 2025-08-12 11:15:00 | 1324.60 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-08-11 14:15:00 | 1273.20 | 2025-08-12 11:15:00 | 1324.60 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2025-08-26 13:45:00 | 1422.60 | 2025-08-28 14:15:00 | 1401.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-08-28 11:00:00 | 1423.20 | 2025-08-28 14:15:00 | 1401.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-09-23 14:45:00 | 1453.50 | 2025-09-26 09:15:00 | 1380.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 1453.50 | 2025-09-26 09:15:00 | 1380.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:45:00 | 1449.10 | 2025-09-26 09:15:00 | 1376.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:45:00 | 1453.50 | 2025-09-30 14:15:00 | 1308.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 1453.50 | 2025-09-30 14:15:00 | 1308.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-24 14:45:00 | 1449.10 | 2025-10-01 09:15:00 | 1304.19 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-07 11:00:00 | 1406.00 | 2025-10-15 10:15:00 | 1546.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1425.00 | 2025-10-15 10:15:00 | 1547.70 | TARGET_HIT | 1.00 | 8.61% |
| BUY | retest2 | 2025-10-08 10:45:00 | 1407.00 | 2025-10-15 10:15:00 | 1548.80 | TARGET_HIT | 1.00 | 10.08% |
| BUY | retest2 | 2025-10-09 11:15:00 | 1408.00 | 2025-10-16 12:15:00 | 1567.50 | TARGET_HIT | 1.00 | 11.33% |
| BUY | retest2 | 2025-10-14 13:45:00 | 1488.40 | 2025-10-17 12:15:00 | 1637.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 09:30:00 | 1494.40 | 2025-10-17 12:15:00 | 1643.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-19 09:15:00 | 1705.10 | 2025-11-19 09:15:00 | 1720.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-20 12:45:00 | 1727.10 | 2025-11-21 10:15:00 | 1706.90 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-21 13:00:00 | 1723.00 | 2025-11-21 14:15:00 | 1697.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-11-21 14:00:00 | 1719.20 | 2025-11-21 14:15:00 | 1697.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-27 12:30:00 | 1792.60 | 2025-12-01 14:15:00 | 1769.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-27 15:00:00 | 1792.10 | 2025-12-01 14:15:00 | 1769.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-11-28 09:30:00 | 1792.10 | 2025-12-01 14:15:00 | 1769.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest1 | 2025-12-08 09:15:00 | 1693.50 | 2025-12-10 09:15:00 | 1701.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-12-09 14:00:00 | 1675.40 | 2025-12-10 09:15:00 | 1701.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-12-12 11:45:00 | 1654.00 | 2025-12-16 11:15:00 | 1675.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-12-12 13:15:00 | 1656.40 | 2025-12-16 11:15:00 | 1675.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1648.60 | 2025-12-16 11:15:00 | 1675.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-12-30 09:15:00 | 1671.30 | 2025-12-30 12:15:00 | 1652.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-30 10:00:00 | 1663.50 | 2025-12-30 12:15:00 | 1652.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-06 14:15:00 | 1770.90 | 2026-01-08 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-01-07 10:45:00 | 1773.00 | 2026-01-08 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-01-07 11:15:00 | 1770.00 | 2026-01-08 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-07 12:30:00 | 1776.80 | 2026-01-08 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-02-06 09:30:00 | 1990.20 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2026-02-06 12:00:00 | 1990.30 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2026-02-06 12:30:00 | 1996.90 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2026-02-09 09:15:00 | 1999.30 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2026-02-10 09:15:00 | 2023.40 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-02-20 13:15:00 | 2112.30 | 2026-02-27 15:15:00 | 2152.10 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2026-02-20 14:45:00 | 2113.20 | 2026-02-27 15:15:00 | 2152.10 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2026-02-23 09:15:00 | 2142.70 | 2026-02-27 15:15:00 | 2152.10 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2026-03-02 14:45:00 | 2188.40 | 2026-03-04 09:15:00 | 2108.00 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-03-13 09:15:00 | 2154.00 | 2026-03-17 09:15:00 | 2305.80 | STOP_HIT | 1.00 | -7.05% |
| BUY | retest2 | 2026-03-30 15:15:00 | 2572.00 | 2026-04-01 09:15:00 | 2459.20 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2026-04-06 15:15:00 | 2205.00 | 2026-04-08 09:15:00 | 2285.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2026-04-13 12:00:00 | 2348.00 | 2026-04-15 12:15:00 | 2307.40 | STOP_HIT | 1.00 | -1.73% |
