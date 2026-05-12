# Five-Star Business Finance Ltd. (FIVESTAR)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 462.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 49 |
| ALERT2 | 49 |
| ALERT2_SKIP | 30 |
| ALERT3 | 129 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 20 / 35
- **Target hits / Stop hits / Partials:** 2 / 46 / 7
- **Avg / median % per leg:** 0.65% / -0.44%
- **Sum % (uncompounded):** 35.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 2 | 10.5% | 1 | 18 | 0 | -0.45% | -8.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 2 | 10.5% | 1 | 18 | 0 | -0.45% | -8.5% |
| SELL (all) | 36 | 18 | 50.0% | 1 | 28 | 7 | 1.23% | 44.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 18 | 50.0% | 1 | 28 | 7 | 1.23% | 44.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 20 | 36.4% | 2 | 46 | 7 | 0.65% | 35.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 693.75 | 684.11 | 684.01 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 09:15:00 | 679.95 | 686.59 | 686.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 10:15:00 | 675.50 | 679.99 | 682.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 13:15:00 | 679.45 | 679.32 | 681.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 14:00:00 | 679.45 | 679.32 | 681.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 675.00 | 678.36 | 680.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 10:30:00 | 674.70 | 677.68 | 680.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 11:15:00 | 674.90 | 677.68 | 680.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 12:30:00 | 674.55 | 676.59 | 679.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 09:30:00 | 671.50 | 675.95 | 678.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 676.40 | 673.85 | 675.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 679.65 | 673.85 | 675.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 677.00 | 674.48 | 675.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 677.15 | 674.48 | 675.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 677.65 | 675.11 | 675.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 677.15 | 675.11 | 675.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 677.00 | 675.49 | 675.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 677.80 | 675.49 | 675.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 672.30 | 674.85 | 675.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 14:15:00 | 666.10 | 674.85 | 675.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 09:15:00 | 678.80 | 674.40 | 675.08 | SL hit (close>static) qty=1.00 sl=678.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 680.30 | 675.58 | 675.56 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 671.35 | 674.84 | 675.23 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 690.95 | 677.48 | 676.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 695.00 | 688.42 | 684.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 14:15:00 | 707.60 | 708.85 | 702.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 14:15:00 | 707.60 | 708.85 | 702.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 707.60 | 708.85 | 702.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 707.60 | 708.85 | 702.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 704.85 | 707.59 | 703.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:15:00 | 702.05 | 707.59 | 703.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 706.05 | 707.28 | 703.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 704.40 | 707.28 | 703.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 695.70 | 704.97 | 702.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 695.70 | 704.97 | 702.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 696.90 | 703.35 | 702.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 695.80 | 703.35 | 702.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 688.15 | 698.86 | 700.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 684.15 | 694.82 | 698.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 686.45 | 684.77 | 690.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:15:00 | 685.85 | 684.77 | 690.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 681.80 | 684.17 | 689.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:30:00 | 680.10 | 684.72 | 689.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 696.45 | 687.07 | 690.14 | SL hit (close>static) qty=1.00 sl=691.60 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 708.60 | 694.96 | 693.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 720.80 | 702.21 | 697.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 14:15:00 | 701.00 | 709.51 | 703.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 701.00 | 709.51 | 703.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 701.00 | 709.51 | 703.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 701.00 | 709.51 | 703.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 702.00 | 708.01 | 703.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 688.40 | 708.01 | 703.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 685.60 | 703.53 | 701.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 686.80 | 703.53 | 701.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 682.15 | 699.25 | 699.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 681.00 | 692.91 | 696.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 686.55 | 686.34 | 691.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:00:00 | 686.55 | 686.34 | 691.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 687.85 | 686.70 | 689.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 688.20 | 686.70 | 689.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 688.05 | 687.00 | 689.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 688.05 | 687.00 | 689.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 687.80 | 687.16 | 688.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:00:00 | 687.80 | 687.16 | 688.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 690.70 | 687.87 | 689.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 690.70 | 687.87 | 689.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 691.20 | 688.53 | 689.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 689.85 | 688.53 | 689.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 694.30 | 689.90 | 689.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 694.30 | 689.90 | 689.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 709.00 | 693.72 | 691.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 796.75 | 799.46 | 787.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 09:45:00 | 794.15 | 799.46 | 787.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 787.50 | 795.96 | 788.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 787.60 | 795.96 | 788.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 787.20 | 794.21 | 788.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:30:00 | 786.85 | 794.21 | 788.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 775.60 | 790.49 | 786.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 775.60 | 790.49 | 786.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 780.70 | 788.53 | 786.32 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 775.25 | 784.25 | 784.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 766.30 | 776.96 | 780.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 14:15:00 | 757.00 | 753.62 | 760.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 14:15:00 | 757.00 | 753.62 | 760.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 757.00 | 753.62 | 760.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 757.00 | 753.62 | 760.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 756.30 | 753.48 | 758.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 756.30 | 753.48 | 758.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 758.90 | 754.56 | 758.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 762.95 | 754.56 | 758.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 764.00 | 756.45 | 759.40 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 768.00 | 761.62 | 761.26 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 758.50 | 760.94 | 761.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 756.25 | 760.00 | 760.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 750.10 | 748.82 | 753.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 750.10 | 748.82 | 753.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 750.10 | 748.82 | 753.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:45:00 | 734.60 | 746.38 | 751.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 15:15:00 | 745.95 | 740.03 | 739.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 15:15:00 | 745.95 | 740.03 | 739.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 750.70 | 743.91 | 741.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 14:15:00 | 771.10 | 772.76 | 765.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 771.10 | 772.76 | 765.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 757.60 | 769.29 | 765.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 757.60 | 769.29 | 765.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 760.90 | 767.62 | 764.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 762.85 | 766.66 | 764.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:30:00 | 762.95 | 765.26 | 764.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 768.10 | 765.10 | 764.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 762.90 | 764.36 | 764.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 762.90 | 764.36 | 764.53 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 771.50 | 765.79 | 765.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 779.85 | 768.60 | 766.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 770.10 | 771.16 | 768.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 14:15:00 | 770.10 | 771.16 | 768.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 770.10 | 771.16 | 768.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 770.10 | 771.16 | 768.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 768.80 | 770.69 | 768.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 769.50 | 770.69 | 768.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 774.70 | 771.49 | 769.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:45:00 | 779.45 | 772.73 | 769.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 775.00 | 771.48 | 770.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 766.45 | 770.18 | 770.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 766.45 | 770.18 | 770.32 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 776.15 | 771.37 | 770.85 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 760.50 | 770.08 | 770.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 757.20 | 767.51 | 769.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 752.20 | 751.30 | 755.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 752.20 | 751.30 | 755.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 761.70 | 753.38 | 756.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 761.70 | 753.38 | 756.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 758.60 | 754.42 | 756.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:30:00 | 762.50 | 754.42 | 756.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 760.00 | 756.11 | 756.80 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 762.05 | 758.17 | 757.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 12:15:00 | 768.85 | 760.31 | 758.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 770.00 | 774.51 | 769.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 770.00 | 774.51 | 769.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 770.00 | 774.51 | 769.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 770.00 | 774.51 | 769.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 768.10 | 773.23 | 769.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:30:00 | 768.10 | 773.23 | 769.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 756.55 | 769.89 | 768.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 756.55 | 769.89 | 768.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 12:15:00 | 751.15 | 766.14 | 766.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 13:15:00 | 748.75 | 762.66 | 765.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 749.55 | 748.99 | 754.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 749.55 | 748.99 | 754.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 749.55 | 748.99 | 754.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:00:00 | 745.00 | 748.19 | 753.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 707.75 | 717.57 | 722.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-28 13:15:00 | 670.50 | 682.63 | 696.82 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 580.00 | 578.70 | 578.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 585.00 | 579.96 | 579.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 589.60 | 592.98 | 589.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 589.60 | 592.98 | 589.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 589.60 | 592.98 | 589.04 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 580.25 | 586.93 | 587.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 11:15:00 | 578.80 | 585.31 | 586.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 584.80 | 582.41 | 584.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 584.80 | 582.41 | 584.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 584.80 | 582.41 | 584.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 584.80 | 582.41 | 584.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 585.00 | 582.93 | 584.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 582.45 | 583.90 | 584.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 587.85 | 582.80 | 583.39 | SL hit (close>static) qty=1.00 sl=586.90 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 584.30 | 583.81 | 583.78 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 583.30 | 583.71 | 583.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 13:15:00 | 580.60 | 583.09 | 583.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 550.70 | 549.36 | 555.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 550.55 | 549.36 | 555.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 552.00 | 548.14 | 551.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 543.65 | 548.14 | 551.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 544.90 | 547.49 | 551.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 10:45:00 | 541.75 | 546.22 | 550.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 535.65 | 533.90 | 533.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 535.65 | 533.90 | 533.80 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 531.40 | 533.40 | 533.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 14:15:00 | 529.60 | 532.18 | 532.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 535.15 | 532.52 | 532.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 535.15 | 532.52 | 532.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 535.15 | 532.52 | 532.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 536.35 | 532.52 | 532.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 536.50 | 533.32 | 533.27 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 529.90 | 533.15 | 533.52 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 541.00 | 534.07 | 533.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 542.60 | 538.44 | 536.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 10:15:00 | 538.20 | 538.39 | 536.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:45:00 | 537.65 | 538.39 | 536.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 544.80 | 548.02 | 544.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 544.80 | 548.02 | 544.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 541.60 | 546.73 | 544.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 541.60 | 546.73 | 544.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 544.20 | 546.23 | 544.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 542.00 | 546.23 | 544.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 542.00 | 544.90 | 544.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 542.00 | 544.90 | 544.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 537.95 | 543.51 | 543.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 536.15 | 539.76 | 541.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 15:15:00 | 538.75 | 537.86 | 539.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:15:00 | 537.65 | 537.86 | 539.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 535.35 | 537.36 | 539.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 532.45 | 536.57 | 537.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:45:00 | 528.20 | 527.51 | 531.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 505.83 | 513.13 | 517.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 501.79 | 513.13 | 517.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 515.00 | 511.70 | 515.04 | SL hit (close>ema200) qty=0.50 sl=511.70 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 532.50 | 517.19 | 517.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 10:15:00 | 538.35 | 521.42 | 518.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 533.70 | 534.01 | 530.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 10:30:00 | 533.75 | 534.01 | 530.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 531.50 | 533.18 | 530.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 532.85 | 533.05 | 530.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 532.75 | 534.32 | 534.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 530.60 | 533.58 | 533.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 530.60 | 533.58 | 533.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 526.45 | 532.15 | 533.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 522.40 | 522.19 | 525.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 522.40 | 522.19 | 525.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 525.65 | 523.55 | 525.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 524.90 | 523.55 | 525.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 523.25 | 523.49 | 525.36 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 529.10 | 526.86 | 526.57 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 525.15 | 526.41 | 526.41 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 528.05 | 526.73 | 526.56 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 522.00 | 525.73 | 526.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 521.60 | 524.91 | 525.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 525.75 | 525.08 | 525.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 525.75 | 525.08 | 525.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 525.75 | 525.08 | 525.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 525.75 | 525.08 | 525.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 519.35 | 523.71 | 524.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 517.50 | 523.71 | 524.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:00:00 | 518.00 | 522.57 | 524.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 531.85 | 520.56 | 522.01 | SL hit (close>static) qty=1.00 sl=528.20 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 528.65 | 523.73 | 523.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 531.70 | 527.71 | 525.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 523.45 | 529.80 | 528.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 523.45 | 529.80 | 528.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 523.45 | 529.80 | 528.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 523.45 | 529.80 | 528.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 525.55 | 528.95 | 527.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 527.15 | 528.95 | 527.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 12:00:00 | 526.85 | 528.53 | 527.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 525.40 | 527.41 | 527.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 525.40 | 527.41 | 527.42 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 527.80 | 527.49 | 527.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 532.25 | 528.44 | 527.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 531.95 | 532.33 | 530.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 531.95 | 532.33 | 530.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 535.80 | 533.44 | 531.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 535.80 | 533.44 | 531.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 535.40 | 533.83 | 531.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 532.20 | 533.83 | 531.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 533.25 | 535.74 | 534.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 533.25 | 535.74 | 534.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 535.00 | 535.59 | 534.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 538.05 | 535.99 | 534.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 538.10 | 536.41 | 535.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 538.50 | 536.23 | 535.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 537.00 | 536.30 | 535.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 531.90 | 535.42 | 535.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 531.90 | 535.42 | 535.03 | SL hit (close<static) qty=1.00 sl=532.15 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 526.35 | 533.60 | 534.24 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 537.00 | 533.57 | 533.36 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 531.10 | 533.23 | 533.40 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 543.00 | 535.18 | 534.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 584.10 | 544.97 | 538.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 650.10 | 653.15 | 639.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 650.10 | 653.15 | 639.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 650.10 | 653.15 | 639.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 653.55 | 653.15 | 639.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 630.55 | 645.14 | 641.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 630.55 | 645.14 | 641.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 625.55 | 641.22 | 640.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 618.10 | 641.22 | 640.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 626.10 | 638.20 | 638.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 622.20 | 631.16 | 634.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 637.80 | 632.49 | 635.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 637.80 | 632.49 | 635.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 637.80 | 632.49 | 635.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 637.80 | 632.49 | 635.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 648.40 | 635.67 | 636.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 646.75 | 635.67 | 636.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 648.25 | 638.19 | 637.46 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 631.90 | 638.01 | 638.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 624.30 | 633.44 | 636.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 13:15:00 | 619.85 | 619.65 | 625.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 13:45:00 | 620.40 | 619.65 | 625.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 628.45 | 620.18 | 624.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 628.45 | 620.18 | 624.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 630.05 | 622.15 | 624.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 630.05 | 622.15 | 624.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 636.00 | 624.92 | 625.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:00:00 | 636.00 | 624.92 | 625.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 635.85 | 627.11 | 626.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 13:15:00 | 642.45 | 630.17 | 628.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 643.50 | 643.65 | 637.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 643.50 | 643.65 | 637.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 635.50 | 641.73 | 637.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 635.25 | 641.73 | 637.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 635.15 | 640.41 | 637.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 635.50 | 640.41 | 637.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 13:15:00 | 628.50 | 635.25 | 635.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 622.85 | 632.02 | 633.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 10:15:00 | 599.05 | 598.60 | 606.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 11:00:00 | 599.05 | 598.60 | 606.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 607.70 | 600.42 | 606.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 607.70 | 600.42 | 606.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 600.90 | 600.52 | 606.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:45:00 | 602.60 | 600.52 | 606.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 617.15 | 603.84 | 607.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 617.15 | 603.84 | 607.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 620.00 | 607.08 | 608.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 620.00 | 607.08 | 608.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 15:15:00 | 619.85 | 609.63 | 609.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 10:15:00 | 624.55 | 614.35 | 611.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 612.15 | 615.82 | 613.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 13:15:00 | 612.15 | 615.82 | 613.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 612.15 | 615.82 | 613.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 612.15 | 615.82 | 613.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 617.00 | 616.06 | 613.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:30:00 | 613.45 | 616.06 | 613.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 611.20 | 615.09 | 613.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 599.20 | 615.09 | 613.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 602.30 | 612.53 | 612.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 601.50 | 612.53 | 612.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 601.60 | 610.34 | 611.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 09:15:00 | 596.15 | 605.27 | 607.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 620.75 | 599.87 | 602.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 620.75 | 599.87 | 602.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 620.75 | 599.87 | 602.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:45:00 | 623.50 | 599.87 | 602.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 602.05 | 600.31 | 602.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:30:00 | 596.25 | 598.68 | 601.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 592.50 | 584.35 | 583.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 592.50 | 584.35 | 583.58 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 577.50 | 583.00 | 583.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 572.55 | 579.47 | 581.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 576.00 | 572.40 | 575.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 576.00 | 572.40 | 575.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 576.00 | 572.40 | 575.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 576.00 | 572.40 | 575.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 576.35 | 573.19 | 575.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 576.55 | 573.19 | 575.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 578.90 | 574.33 | 575.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 578.90 | 574.33 | 575.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 577.55 | 574.98 | 575.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 574.85 | 574.98 | 575.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 574.15 | 567.68 | 567.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 574.15 | 567.68 | 567.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 11:15:00 | 584.75 | 575.73 | 572.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 586.15 | 586.46 | 580.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 586.15 | 586.46 | 580.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 586.15 | 586.46 | 580.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:15:00 | 591.75 | 586.42 | 580.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 13:15:00 | 590.60 | 586.56 | 581.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 15:00:00 | 590.50 | 588.18 | 583.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 15:00:00 | 590.05 | 589.53 | 586.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 583.95 | 588.55 | 586.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 583.95 | 588.55 | 586.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 584.95 | 587.83 | 586.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:15:00 | 581.50 | 587.83 | 586.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 580.00 | 586.26 | 585.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 579.30 | 586.26 | 585.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 579.05 | 584.82 | 585.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 579.05 | 584.82 | 585.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 13:15:00 | 569.75 | 581.81 | 583.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 571.00 | 565.23 | 571.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 571.00 | 565.23 | 571.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 571.00 | 565.23 | 571.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 571.65 | 565.23 | 571.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 569.10 | 566.01 | 571.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 13:30:00 | 564.55 | 566.54 | 570.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 15:00:00 | 564.40 | 566.11 | 569.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:15:00 | 536.32 | 543.15 | 549.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:15:00 | 536.18 | 543.15 | 549.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 15:15:00 | 540.45 | 540.25 | 545.03 | SL hit (close>ema200) qty=0.50 sl=540.25 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 517.85 | 512.61 | 512.61 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 508.00 | 514.71 | 514.78 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 14:15:00 | 516.70 | 511.28 | 511.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 521.40 | 514.22 | 512.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 13:15:00 | 512.50 | 515.37 | 513.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 13:15:00 | 512.50 | 515.37 | 513.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 512.50 | 515.37 | 513.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 512.35 | 515.37 | 513.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 515.00 | 515.30 | 513.88 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 508.60 | 513.05 | 513.15 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 516.95 | 512.23 | 512.16 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 507.85 | 511.36 | 511.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 11:15:00 | 505.80 | 510.03 | 511.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 467.25 | 462.20 | 477.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:00:00 | 467.25 | 462.20 | 477.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 452.60 | 442.17 | 449.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:00:00 | 447.90 | 446.41 | 449.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:30:00 | 447.35 | 446.61 | 449.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 445.55 | 447.03 | 449.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 457.95 | 450.51 | 450.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 457.95 | 450.51 | 450.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 464.90 | 454.30 | 452.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 449.50 | 454.98 | 453.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 449.50 | 454.98 | 453.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 449.50 | 454.98 | 453.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 445.00 | 454.98 | 453.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 447.40 | 453.47 | 452.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 447.40 | 453.47 | 452.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 444.40 | 451.65 | 451.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 12:15:00 | 442.00 | 449.72 | 451.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 448.45 | 446.77 | 448.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 448.45 | 446.77 | 448.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 448.45 | 446.77 | 448.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 448.45 | 446.77 | 448.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 449.75 | 447.36 | 448.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 449.75 | 447.36 | 448.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 451.80 | 448.25 | 448.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 450.30 | 448.25 | 448.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 447.45 | 448.24 | 448.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 454.45 | 448.24 | 448.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 442.25 | 447.04 | 448.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:45:00 | 448.00 | 447.04 | 448.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 440.15 | 442.82 | 445.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:45:00 | 441.85 | 442.82 | 445.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 436.50 | 433.63 | 437.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 440.60 | 433.63 | 437.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 452.90 | 437.48 | 438.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 452.90 | 437.48 | 438.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 10:15:00 | 453.00 | 440.58 | 440.15 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 439.90 | 440.87 | 440.92 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 442.70 | 440.60 | 440.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 445.15 | 442.07 | 441.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 439.50 | 443.13 | 442.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 439.50 | 443.13 | 442.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 439.50 | 443.13 | 442.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 439.50 | 443.13 | 442.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 443.15 | 443.13 | 442.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 443.60 | 443.40 | 442.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 438.00 | 441.53 | 441.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 15:15:00 | 438.00 | 441.53 | 441.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 436.65 | 440.56 | 441.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 11:15:00 | 442.00 | 440.16 | 441.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 442.00 | 440.16 | 441.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 442.00 | 440.16 | 441.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 442.00 | 440.16 | 441.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 445.65 | 441.26 | 441.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 445.65 | 441.26 | 441.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 445.00 | 442.00 | 441.78 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 439.45 | 441.47 | 441.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 438.00 | 440.78 | 441.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 428.75 | 427.13 | 429.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 428.75 | 427.13 | 429.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 428.75 | 427.13 | 429.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 424.30 | 426.46 | 429.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 423.45 | 425.79 | 428.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 403.08 | 416.40 | 420.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 402.28 | 416.40 | 420.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 403.95 | 402.62 | 406.62 | SL hit (close>ema200) qty=0.50 sl=402.62 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 373.40 | 359.82 | 359.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 375.60 | 362.98 | 360.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 365.00 | 369.86 | 365.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 365.00 | 369.86 | 365.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 365.00 | 369.86 | 365.84 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 354.85 | 362.87 | 363.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 352.20 | 357.75 | 360.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 349.00 | 348.97 | 353.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:00:00 | 349.00 | 348.97 | 353.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 356.55 | 350.14 | 353.01 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 371.50 | 357.78 | 356.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 384.15 | 365.64 | 360.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 376.25 | 381.19 | 375.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 13:15:00 | 376.25 | 381.19 | 375.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 376.25 | 381.19 | 375.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 376.25 | 381.19 | 375.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 374.45 | 379.84 | 375.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:30:00 | 374.15 | 379.84 | 375.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 375.00 | 378.87 | 375.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 363.95 | 378.87 | 375.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 364.70 | 376.04 | 374.55 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 364.00 | 371.72 | 372.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 359.45 | 369.27 | 371.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 365.85 | 362.50 | 367.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 365.85 | 362.50 | 367.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 365.85 | 362.50 | 367.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 365.05 | 362.50 | 367.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 365.35 | 363.07 | 366.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 363.65 | 364.10 | 366.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 369.05 | 359.80 | 360.12 | SL hit (close>static) qty=1.00 sl=366.90 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 366.10 | 361.06 | 360.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 376.00 | 367.43 | 364.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 15:15:00 | 423.20 | 426.56 | 415.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 09:15:00 | 412.60 | 426.56 | 415.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 412.75 | 423.80 | 415.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 418.65 | 423.80 | 415.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 10:15:00 | 460.51 | 447.71 | 436.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 485.05 | 489.71 | 490.22 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 495.45 | 490.29 | 489.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 500.65 | 492.39 | 490.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 501.40 | 504.44 | 498.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 09:45:00 | 501.80 | 504.44 | 498.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 499.15 | 504.83 | 501.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 499.15 | 504.83 | 501.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 501.00 | 504.06 | 501.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 482.15 | 504.06 | 501.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 494.05 | 502.06 | 500.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 475.65 | 502.06 | 500.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 495.05 | 499.53 | 499.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 492.45 | 497.31 | 498.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 14:15:00 | 498.25 | 497.50 | 498.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 498.25 | 497.50 | 498.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 498.25 | 497.50 | 498.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 498.25 | 497.50 | 498.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 502.00 | 498.40 | 498.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 502.30 | 498.40 | 498.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 493.85 | 497.49 | 498.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 12:00:00 | 489.10 | 495.09 | 497.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:30:00 | 486.85 | 480.51 | 482.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 489.00 | 483.24 | 483.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 489.00 | 483.24 | 483.09 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 471.00 | 480.79 | 481.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 10:15:00 | 470.80 | 478.79 | 480.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 13:15:00 | 477.90 | 477.10 | 479.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 13:15:00 | 477.90 | 477.10 | 479.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 477.90 | 477.10 | 479.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 479.45 | 477.10 | 479.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 473.50 | 476.38 | 478.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:30:00 | 465.10 | 473.89 | 477.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 10:30:00 | 674.70 | 2025-05-21 09:15:00 | 678.80 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-05-16 11:15:00 | 674.90 | 2025-05-21 10:15:00 | 680.30 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-05-16 12:30:00 | 674.55 | 2025-05-21 10:15:00 | 680.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-05-19 09:30:00 | 671.50 | 2025-05-21 10:15:00 | 680.30 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-20 14:15:00 | 666.10 | 2025-05-21 10:15:00 | 680.30 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-05-30 10:30:00 | 680.10 | 2025-05-30 11:15:00 | 696.45 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-06-05 15:15:00 | 689.85 | 2025-06-06 09:15:00 | 694.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-06-20 13:45:00 | 734.60 | 2025-06-24 15:15:00 | 745.95 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-07-01 12:00:00 | 762.85 | 2025-07-02 15:15:00 | 762.90 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-01 13:30:00 | 762.95 | 2025-07-02 15:15:00 | 762.90 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-07-02 14:15:00 | 768.10 | 2025-07-02 15:15:00 | 762.90 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-04 10:45:00 | 779.45 | 2025-07-07 13:15:00 | 766.45 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-07 09:15:00 | 775.00 | 2025-07-07 13:15:00 | 766.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-07-17 11:00:00 | 745.00 | 2025-07-25 10:15:00 | 707.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 11:00:00 | 745.00 | 2025-07-28 13:15:00 | 670.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-22 09:15:00 | 582.45 | 2025-08-22 14:15:00 | 587.85 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-08-25 09:15:00 | 583.30 | 2025-08-25 11:15:00 | 584.30 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-08-25 09:45:00 | 583.25 | 2025-08-25 11:15:00 | 584.30 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-09-02 10:45:00 | 541.75 | 2025-09-09 10:15:00 | 535.65 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2025-09-23 09:15:00 | 532.45 | 2025-09-29 09:15:00 | 505.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:45:00 | 528.20 | 2025-09-29 09:15:00 | 501.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:15:00 | 532.45 | 2025-09-29 14:15:00 | 515.00 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-09-24 09:45:00 | 528.20 | 2025-09-29 14:15:00 | 515.00 | STOP_HIT | 0.50 | 2.50% |
| BUY | retest2 | 2025-10-03 14:15:00 | 532.85 | 2025-10-08 09:15:00 | 530.60 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-08 09:15:00 | 532.75 | 2025-10-08 09:15:00 | 530.60 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-10-14 10:15:00 | 517.50 | 2025-10-15 09:15:00 | 531.85 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-10-14 11:00:00 | 518.00 | 2025-10-15 09:15:00 | 531.85 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-10-17 11:15:00 | 527.15 | 2025-10-17 13:15:00 | 525.40 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-10-17 12:00:00 | 526.85 | 2025-10-17 13:15:00 | 525.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-24 14:00:00 | 538.05 | 2025-10-27 11:15:00 | 531.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-24 15:00:00 | 538.10 | 2025-10-27 11:15:00 | 531.90 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-27 09:15:00 | 538.50 | 2025-10-27 11:15:00 | 531.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-10-27 10:45:00 | 537.00 | 2025-10-27 11:15:00 | 531.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-28 11:30:00 | 596.25 | 2025-12-04 13:15:00 | 592.50 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-12-10 09:15:00 | 574.85 | 2025-12-15 09:15:00 | 574.15 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-12-17 11:15:00 | 591.75 | 2025-12-19 12:15:00 | 579.05 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-12-17 13:15:00 | 590.60 | 2025-12-19 12:15:00 | 579.05 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-12-17 15:00:00 | 590.50 | 2025-12-19 12:15:00 | 579.05 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-12-18 15:00:00 | 590.05 | 2025-12-19 12:15:00 | 579.05 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-12-23 13:30:00 | 564.55 | 2025-12-30 09:15:00 | 536.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 15:00:00 | 564.40 | 2025-12-30 09:15:00 | 536.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 13:30:00 | 564.55 | 2025-12-30 15:15:00 | 540.45 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2025-12-23 15:00:00 | 564.40 | 2025-12-30 15:15:00 | 540.45 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2026-02-03 14:00:00 | 447.90 | 2026-02-04 11:15:00 | 457.95 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-02-03 14:30:00 | 447.35 | 2026-02-04 11:15:00 | 457.95 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-04 09:15:00 | 445.55 | 2026-02-04 11:15:00 | 457.95 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-02-18 11:45:00 | 443.60 | 2026-02-18 15:15:00 | 438.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-26 11:30:00 | 424.30 | 2026-03-02 09:15:00 | 403.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:30:00 | 423.45 | 2026-03-02 09:15:00 | 402.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 424.30 | 2026-03-05 11:15:00 | 403.95 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2026-02-26 12:30:00 | 423.45 | 2026-03-05 11:15:00 | 403.95 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2026-04-01 13:15:00 | 363.65 | 2026-04-07 09:15:00 | 369.05 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-04-13 10:15:00 | 418.65 | 2026-04-16 10:15:00 | 460.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 12:00:00 | 489.10 | 2026-05-06 15:15:00 | 489.00 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2026-05-06 09:30:00 | 486.85 | 2026-05-06 15:15:00 | 489.00 | STOP_HIT | 1.00 | -0.44% |
