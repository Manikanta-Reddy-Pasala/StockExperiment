# Computer Age Management Services Ltd. (CAMS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 835.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 148 |
| ALERT1 | 106 |
| ALERT2 | 104 |
| ALERT2_SKIP | 52 |
| ALERT3 | 286 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 121 |
| PARTIAL | 15 |
| TARGET_HIT | 8 |
| STOP_HIT | 117 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 92
- **Target hits / Stop hits / Partials:** 8 / 117 / 15
- **Avg / median % per leg:** 0.57% / -0.97%
- **Sum % (uncompounded):** 79.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 18 | 27.7% | 4 | 60 | 1 | 0.47% | 30.3% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | 1.29% | 6.5% |
| BUY @ 3rd Alert (retest2) | 60 | 16 | 26.7% | 4 | 56 | 0 | 0.40% | 23.8% |
| SELL (all) | 75 | 30 | 40.0% | 4 | 57 | 14 | 0.65% | 48.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 75 | 30 | 40.0% | 4 | 57 | 14 | 0.65% | 48.9% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 1.29% | 6.5% |
| retest2 (combined) | 135 | 46 | 34.1% | 8 | 113 | 14 | 0.54% | 72.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 615.71 | 635.11 | 636.96 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 639.18 | 632.47 | 632.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 659.60 | 638.83 | 635.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 654.96 | 655.63 | 649.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 654.96 | 655.63 | 649.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 700.14 | 707.37 | 700.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 700.14 | 707.37 | 700.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 698.77 | 705.65 | 700.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:30:00 | 697.55 | 705.65 | 700.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 696.31 | 703.78 | 699.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:45:00 | 696.00 | 703.78 | 699.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 685.00 | 696.61 | 696.98 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 10:15:00 | 709.94 | 699.39 | 698.19 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 693.60 | 699.65 | 699.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 669.72 | 690.92 | 695.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 688.00 | 684.92 | 690.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 14:00:00 | 688.00 | 684.92 | 690.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 690.39 | 686.02 | 690.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 690.39 | 686.02 | 690.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 687.12 | 686.24 | 690.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 686.62 | 686.24 | 690.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 680.10 | 685.01 | 689.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 10:30:00 | 678.51 | 683.73 | 688.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 15:15:00 | 696.00 | 687.38 | 688.30 | SL hit (close>static) qty=1.00 sl=691.80 alert=retest2 |

### Cycle 6 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 696.41 | 689.18 | 689.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 720.96 | 701.03 | 695.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 15:15:00 | 708.07 | 709.76 | 703.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 09:15:00 | 677.30 | 709.76 | 703.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 695.08 | 706.83 | 702.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 680.51 | 706.83 | 702.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 668.20 | 699.10 | 699.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 647.70 | 688.82 | 694.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 671.29 | 671.17 | 682.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:00:00 | 671.29 | 671.17 | 682.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 684.76 | 673.89 | 682.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 684.76 | 673.89 | 682.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 686.00 | 676.31 | 682.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 684.13 | 676.31 | 682.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 690.24 | 679.10 | 683.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 690.24 | 679.10 | 683.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 687.79 | 681.46 | 683.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 685.63 | 681.46 | 683.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 687.24 | 682.62 | 684.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 700.00 | 682.62 | 684.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 701.80 | 686.45 | 685.73 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-06 14:15:00 | 674.00 | 683.89 | 685.08 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 11:15:00 | 692.00 | 684.37 | 683.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 12:15:00 | 695.42 | 686.58 | 685.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 686.68 | 688.00 | 686.19 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 697.00 | 688.00 | 686.19 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 687.93 | 692.15 | 689.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-11 14:15:00 | 687.93 | 692.15 | 689.60 | SL hit (close<ema400) qty=1.00 sl=689.60 alert=retest1 |

### Cycle 11 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 709.98 | 716.89 | 717.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 14:15:00 | 704.82 | 712.75 | 715.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 717.75 | 712.35 | 714.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 717.75 | 712.35 | 714.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 717.75 | 712.35 | 714.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:45:00 | 722.60 | 712.35 | 714.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 717.29 | 713.34 | 714.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 716.13 | 713.34 | 714.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 718.79 | 714.43 | 715.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:00:00 | 718.79 | 714.43 | 715.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 13:15:00 | 719.22 | 715.88 | 715.81 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 708.80 | 714.46 | 715.17 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 725.19 | 716.53 | 715.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 14:15:00 | 731.40 | 721.17 | 718.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 14:15:00 | 736.76 | 738.64 | 730.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 15:00:00 | 736.76 | 738.64 | 730.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 732.00 | 737.21 | 731.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:30:00 | 732.05 | 737.21 | 731.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 731.90 | 736.15 | 731.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 731.90 | 736.15 | 731.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 732.38 | 735.40 | 731.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:30:00 | 731.60 | 735.40 | 731.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 729.99 | 734.31 | 731.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:45:00 | 729.42 | 734.31 | 731.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 729.24 | 733.30 | 731.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:00:00 | 729.24 | 733.30 | 731.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 730.28 | 732.70 | 731.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 730.28 | 732.70 | 731.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 728.00 | 731.76 | 730.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 727.06 | 731.76 | 730.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 726.20 | 730.65 | 730.50 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 722.80 | 729.08 | 729.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 11:15:00 | 720.20 | 727.30 | 728.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 11:15:00 | 722.20 | 717.84 | 721.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 11:15:00 | 722.20 | 717.84 | 721.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 722.20 | 717.84 | 721.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:45:00 | 720.58 | 717.84 | 721.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 723.40 | 718.95 | 722.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 723.40 | 718.95 | 722.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 729.01 | 720.96 | 722.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 729.96 | 720.96 | 722.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 724.39 | 721.65 | 722.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:45:00 | 724.94 | 721.65 | 722.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 719.98 | 721.31 | 722.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 728.97 | 721.31 | 722.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 733.00 | 723.65 | 723.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 744.00 | 732.52 | 728.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 744.40 | 754.37 | 748.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 10:15:00 | 744.40 | 754.37 | 748.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 744.40 | 754.37 | 748.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:00:00 | 744.40 | 754.37 | 748.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 745.62 | 752.62 | 748.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:15:00 | 748.69 | 747.89 | 747.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 12:15:00 | 751.01 | 752.96 | 751.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-10 09:15:00 | 823.56 | 788.68 | 772.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 816.91 | 827.09 | 828.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 793.60 | 816.92 | 820.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 10:15:00 | 818.46 | 812.64 | 816.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 10:15:00 | 818.46 | 812.64 | 816.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 818.46 | 812.64 | 816.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 820.80 | 812.64 | 816.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 814.40 | 812.99 | 816.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:00:00 | 810.96 | 812.71 | 815.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 808.85 | 813.14 | 815.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:45:00 | 812.00 | 808.95 | 811.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 845.76 | 814.94 | 812.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 845.76 | 814.94 | 812.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 11:15:00 | 849.20 | 826.40 | 818.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 902.72 | 933.32 | 907.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 14:15:00 | 902.72 | 933.32 | 907.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 902.72 | 933.32 | 907.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 902.72 | 933.32 | 907.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 900.10 | 926.68 | 906.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 893.86 | 926.68 | 906.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 881.82 | 899.29 | 899.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 877.60 | 887.02 | 892.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 823.48 | 794.69 | 815.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 823.48 | 794.69 | 815.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 823.48 | 794.69 | 815.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:00:00 | 823.48 | 794.69 | 815.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 834.01 | 802.55 | 816.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:30:00 | 816.74 | 811.59 | 818.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 862.68 | 828.18 | 824.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 862.68 | 828.18 | 824.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 886.97 | 857.59 | 844.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 847.21 | 865.64 | 856.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 847.21 | 865.64 | 856.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 847.21 | 865.64 | 856.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 847.21 | 865.64 | 856.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 852.69 | 863.05 | 856.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:45:00 | 854.92 | 858.56 | 855.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 13:15:00 | 855.60 | 858.56 | 855.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 857.98 | 855.57 | 854.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 848.74 | 853.43 | 853.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 848.74 | 853.43 | 853.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 838.07 | 849.36 | 851.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 10:15:00 | 836.92 | 836.49 | 843.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 11:00:00 | 836.92 | 836.49 | 843.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 836.34 | 836.46 | 842.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:30:00 | 840.64 | 836.46 | 842.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 846.03 | 836.91 | 840.44 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 858.57 | 844.50 | 843.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 15:15:00 | 864.00 | 851.35 | 847.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 891.13 | 896.28 | 880.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:45:00 | 890.85 | 896.28 | 880.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 881.67 | 890.83 | 882.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 881.67 | 890.83 | 882.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 883.20 | 889.31 | 882.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 887.08 | 889.31 | 882.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:45:00 | 893.59 | 890.75 | 883.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 12:30:00 | 885.88 | 888.95 | 887.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 14:15:00 | 876.39 | 885.08 | 886.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 876.39 | 885.08 | 886.02 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 15:15:00 | 890.00 | 885.67 | 885.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 905.55 | 889.65 | 887.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 13:15:00 | 891.65 | 892.66 | 889.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 14:00:00 | 891.65 | 892.66 | 889.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 881.73 | 890.48 | 888.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 881.73 | 890.48 | 888.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 885.00 | 889.38 | 888.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 877.88 | 889.38 | 888.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 879.06 | 887.32 | 887.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 11:15:00 | 875.90 | 883.93 | 886.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 872.99 | 871.26 | 875.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 872.99 | 871.26 | 875.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 872.99 | 871.26 | 875.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:30:00 | 869.06 | 869.94 | 874.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 09:30:00 | 864.55 | 867.95 | 871.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:00:00 | 868.62 | 868.08 | 871.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:30:00 | 866.10 | 865.25 | 867.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 873.00 | 866.80 | 867.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:30:00 | 874.66 | 866.80 | 867.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-02 14:15:00 | 883.19 | 870.96 | 869.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 14:15:00 | 883.19 | 870.96 | 869.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 15:15:00 | 886.00 | 873.96 | 871.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 13:15:00 | 874.60 | 875.59 | 873.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 14:00:00 | 874.60 | 875.59 | 873.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 875.12 | 875.50 | 873.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 882.62 | 875.92 | 873.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 871.80 | 881.13 | 881.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 871.80 | 881.13 | 881.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 867.20 | 875.07 | 878.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 09:15:00 | 876.80 | 857.08 | 859.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 876.80 | 857.08 | 859.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 876.80 | 857.08 | 859.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 876.80 | 857.08 | 859.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 875.47 | 860.76 | 861.18 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 873.99 | 863.41 | 862.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 879.11 | 871.34 | 867.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 12:15:00 | 878.40 | 879.03 | 874.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 13:00:00 | 878.40 | 879.03 | 874.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 878.50 | 878.92 | 875.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 15:00:00 | 880.41 | 879.22 | 875.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 12:00:00 | 880.40 | 898.31 | 898.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 12:15:00 | 876.38 | 893.92 | 896.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 876.38 | 893.92 | 896.05 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 935.77 | 901.44 | 897.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 938.42 | 908.84 | 901.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 911.60 | 916.63 | 909.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 911.60 | 916.63 | 909.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 911.60 | 916.63 | 909.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:30:00 | 911.79 | 916.63 | 909.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 910.52 | 915.41 | 909.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:15:00 | 905.20 | 915.41 | 909.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 908.00 | 913.93 | 909.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:30:00 | 909.66 | 913.93 | 909.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 911.44 | 913.43 | 909.73 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 896.74 | 907.95 | 908.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 887.60 | 901.93 | 905.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 12:15:00 | 893.72 | 891.63 | 896.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 12:45:00 | 888.40 | 891.63 | 896.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 898.20 | 892.95 | 896.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:45:00 | 908.95 | 892.95 | 896.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 895.85 | 893.53 | 896.62 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 906.80 | 899.03 | 898.53 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 889.40 | 898.26 | 898.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 886.91 | 895.31 | 897.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 900.12 | 889.61 | 892.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 900.12 | 889.61 | 892.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 900.12 | 889.61 | 892.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 900.12 | 889.61 | 892.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 903.78 | 892.44 | 893.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 903.78 | 892.44 | 893.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 905.97 | 895.15 | 894.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 13:15:00 | 912.00 | 900.32 | 897.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 897.01 | 904.33 | 900.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 897.01 | 904.33 | 900.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 897.01 | 904.33 | 900.15 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 891.29 | 898.00 | 898.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 14:15:00 | 884.45 | 895.29 | 896.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 846.00 | 835.48 | 851.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 846.00 | 835.48 | 851.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 846.00 | 835.48 | 851.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 851.71 | 835.48 | 851.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 862.27 | 840.84 | 852.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 862.27 | 840.84 | 852.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 864.85 | 845.64 | 853.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 864.85 | 845.64 | 853.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 877.16 | 858.46 | 858.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 885.00 | 863.77 | 860.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 13:15:00 | 895.27 | 900.70 | 892.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 13:45:00 | 895.22 | 900.70 | 892.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 883.82 | 897.45 | 892.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 883.58 | 897.45 | 892.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 882.16 | 894.39 | 891.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 881.30 | 894.39 | 891.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 924.68 | 898.54 | 894.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 936.59 | 910.07 | 901.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 947.60 | 921.09 | 911.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 09:30:00 | 959.68 | 962.27 | 953.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 10:15:00 | 942.64 | 949.99 | 950.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 942.64 | 949.99 | 950.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 938.56 | 947.70 | 949.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 903.76 | 902.19 | 917.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 903.76 | 902.19 | 917.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 922.30 | 906.79 | 916.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 922.30 | 906.79 | 916.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 916.84 | 908.80 | 916.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 923.23 | 908.80 | 916.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 911.89 | 909.42 | 916.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 904.07 | 909.31 | 915.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 10:00:00 | 908.40 | 909.31 | 915.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 10:45:00 | 907.01 | 908.49 | 914.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 858.87 | 890.67 | 901.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 862.98 | 890.67 | 901.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 861.66 | 890.67 | 901.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 860.20 | 859.65 | 870.39 | SL hit (close>ema200) qty=0.50 sl=859.65 alert=retest2 |

### Cycle 38 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 882.55 | 871.80 | 871.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 12:15:00 | 894.20 | 879.83 | 876.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 15:15:00 | 908.00 | 910.83 | 900.49 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 09:15:00 | 913.53 | 910.83 | 900.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 903.00 | 910.21 | 902.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 903.00 | 910.21 | 902.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 909.26 | 910.02 | 903.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:45:00 | 915.00 | 910.68 | 904.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:15:00 | 914.13 | 910.68 | 904.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 15:00:00 | 914.06 | 911.35 | 905.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 921.77 | 911.59 | 905.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 949.80 | 939.30 | 933.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:15:00 | 960.60 | 942.91 | 938.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:15:00 | 959.21 | 948.13 | 941.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-12 13:15:00 | 948.84 | 954.51 | 947.25 | SL hit (close<ema200) qty=0.50 sl=954.51 alert=retest1 |

### Cycle 39 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 904.00 | 938.85 | 941.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 896.67 | 916.26 | 928.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 924.63 | 916.14 | 925.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 924.63 | 916.14 | 925.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 924.63 | 916.14 | 925.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 892.40 | 915.55 | 921.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:30:00 | 899.98 | 907.99 | 916.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:45:00 | 900.42 | 906.27 | 914.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 899.11 | 911.32 | 913.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 898.21 | 908.69 | 912.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 09:45:00 | 886.70 | 900.99 | 905.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 907.58 | 901.83 | 901.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 907.58 | 901.83 | 901.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 15:15:00 | 912.02 | 903.87 | 902.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 12:15:00 | 990.00 | 991.29 | 975.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 13:00:00 | 990.00 | 991.29 | 975.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 1047.48 | 1043.49 | 1030.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:30:00 | 1038.26 | 1043.49 | 1030.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1021.86 | 1041.17 | 1033.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 1021.86 | 1041.17 | 1033.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1016.06 | 1036.15 | 1031.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 1016.06 | 1036.15 | 1031.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 12:15:00 | 1019.43 | 1028.97 | 1029.16 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 1041.23 | 1028.68 | 1027.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 14:15:00 | 1056.00 | 1038.71 | 1033.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 1038.83 | 1041.63 | 1035.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 10:00:00 | 1038.83 | 1041.63 | 1035.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1038.00 | 1040.90 | 1035.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 1035.53 | 1040.90 | 1035.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 1038.93 | 1040.48 | 1036.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 1038.50 | 1040.48 | 1036.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1042.79 | 1044.74 | 1041.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:45:00 | 1041.67 | 1044.74 | 1041.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1035.52 | 1049.66 | 1047.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 1035.52 | 1049.66 | 1047.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1036.00 | 1046.93 | 1046.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 1029.59 | 1046.93 | 1046.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 11:15:00 | 1037.92 | 1045.13 | 1045.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 09:15:00 | 1029.00 | 1038.61 | 1042.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 1042.73 | 1038.28 | 1040.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 12:15:00 | 1042.73 | 1038.28 | 1040.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 1042.73 | 1038.28 | 1040.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:30:00 | 1044.00 | 1038.28 | 1040.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 1041.65 | 1038.95 | 1040.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:15:00 | 1038.03 | 1038.95 | 1040.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:45:00 | 1036.20 | 1039.96 | 1041.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 986.13 | 1000.11 | 1012.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 984.39 | 1000.11 | 1012.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 12:15:00 | 983.44 | 982.92 | 993.16 | SL hit (close>ema200) qty=0.50 sl=982.92 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 999.55 | 991.40 | 990.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 11:15:00 | 1001.45 | 993.41 | 991.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 1002.05 | 1005.63 | 1000.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 13:15:00 | 1002.05 | 1005.63 | 1000.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1002.05 | 1005.63 | 1000.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 1002.05 | 1005.63 | 1000.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1011.99 | 1006.90 | 1001.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 15:15:00 | 1020.60 | 1006.90 | 1001.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 975.75 | 1002.86 | 1001.10 | SL hit (close<static) qty=1.00 sl=1001.22 alert=retest2 |

### Cycle 45 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 985.21 | 1015.44 | 1019.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 980.97 | 1003.71 | 1012.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 09:15:00 | 858.71 | 855.90 | 874.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 10:15:00 | 872.78 | 859.28 | 874.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 872.78 | 859.28 | 874.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:00:00 | 872.78 | 859.28 | 874.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 861.04 | 859.63 | 873.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 12:15:00 | 858.86 | 859.63 | 873.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:00:00 | 860.45 | 859.79 | 872.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 901.32 | 868.25 | 871.97 | SL hit (close>static) qty=1.00 sl=876.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 887.10 | 876.07 | 875.13 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 867.08 | 877.43 | 877.58 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 878.73 | 877.78 | 877.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 14:15:00 | 881.60 | 878.55 | 878.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 878.09 | 879.23 | 878.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 878.09 | 879.23 | 878.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 878.09 | 879.23 | 878.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 878.09 | 879.23 | 878.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 884.60 | 880.30 | 879.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:00:00 | 888.44 | 882.38 | 880.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:30:00 | 887.80 | 883.61 | 881.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 891.46 | 883.49 | 881.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 865.98 | 879.99 | 879.98 | SL hit (close<static) qty=1.00 sl=877.92 alert=retest2 |

### Cycle 49 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 852.07 | 874.40 | 877.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 823.77 | 852.38 | 864.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 823.40 | 822.16 | 838.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 823.40 | 822.16 | 838.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 833.39 | 825.79 | 834.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 833.39 | 825.79 | 834.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 834.00 | 827.43 | 834.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 848.43 | 827.43 | 834.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 824.14 | 826.77 | 833.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 817.41 | 826.18 | 830.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 776.54 | 812.85 | 823.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 10:15:00 | 735.67 | 763.53 | 787.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 50 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 740.93 | 710.58 | 706.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 750.07 | 718.48 | 710.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 738.34 | 739.33 | 727.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 738.34 | 739.33 | 727.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 736.24 | 735.20 | 729.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 736.24 | 735.20 | 729.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 743.20 | 736.80 | 731.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 733.45 | 736.80 | 731.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 727.04 | 734.55 | 731.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 727.04 | 734.55 | 731.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 732.53 | 734.15 | 731.54 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 712.00 | 729.70 | 729.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 709.03 | 725.56 | 728.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 698.00 | 684.31 | 695.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 698.00 | 684.31 | 695.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 698.00 | 684.31 | 695.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 698.00 | 684.31 | 695.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 708.84 | 689.22 | 696.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 708.84 | 689.22 | 696.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 701.75 | 691.73 | 696.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 687.07 | 696.32 | 698.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:15:00 | 697.50 | 697.06 | 698.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 695.56 | 696.78 | 698.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:00:00 | 694.12 | 696.78 | 698.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 690.71 | 695.57 | 697.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:30:00 | 696.48 | 695.57 | 697.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 662.62 | 675.35 | 684.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 678.41 | 675.79 | 683.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-14 15:15:00 | 678.41 | 675.79 | 683.11 | SL hit (close>ema200) qty=0.50 sl=675.79 alert=retest2 |

### Cycle 52 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 682.93 | 671.85 | 670.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 696.94 | 678.24 | 674.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 14:15:00 | 686.20 | 690.01 | 685.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 15:00:00 | 686.20 | 690.01 | 685.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 682.82 | 688.57 | 685.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 676.73 | 688.57 | 685.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 678.00 | 686.46 | 684.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 684.27 | 683.63 | 683.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 12:00:00 | 680.38 | 683.63 | 683.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 681.90 | 683.58 | 683.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 681.90 | 683.58 | 683.59 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 14:15:00 | 685.75 | 684.02 | 683.79 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 681.60 | 683.53 | 683.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 676.67 | 682.16 | 682.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 631.01 | 626.44 | 639.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 631.01 | 626.44 | 639.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 629.67 | 627.39 | 631.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 632.90 | 627.39 | 631.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 630.00 | 627.91 | 631.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 630.00 | 627.91 | 631.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 632.74 | 628.88 | 631.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 631.83 | 628.88 | 631.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 633.90 | 629.88 | 631.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:30:00 | 634.65 | 629.88 | 631.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 632.40 | 630.39 | 631.75 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 639.40 | 633.67 | 633.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 654.19 | 637.77 | 635.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 670.03 | 670.69 | 664.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:30:00 | 670.58 | 670.69 | 664.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 672.01 | 671.00 | 665.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:30:00 | 680.10 | 672.68 | 669.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 676.21 | 672.68 | 669.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 14:15:00 | 664.75 | 668.01 | 668.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 14:15:00 | 664.75 | 668.01 | 668.06 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 674.01 | 668.73 | 668.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 12:15:00 | 684.60 | 673.16 | 670.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 13:15:00 | 685.77 | 686.10 | 680.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-17 14:00:00 | 685.77 | 686.10 | 680.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 738.00 | 758.80 | 755.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 738.00 | 758.80 | 755.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 739.40 | 754.92 | 753.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:15:00 | 735.74 | 754.92 | 753.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 735.40 | 751.02 | 752.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 730.20 | 743.05 | 747.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 11:15:00 | 743.87 | 738.73 | 743.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 11:15:00 | 743.87 | 738.73 | 743.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 743.87 | 738.73 | 743.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 743.87 | 738.73 | 743.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 739.00 | 738.78 | 743.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:45:00 | 737.48 | 739.14 | 742.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 15:15:00 | 738.00 | 739.14 | 742.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 10:15:00 | 738.72 | 738.92 | 741.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 10:45:00 | 738.41 | 738.15 | 741.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 737.55 | 737.50 | 739.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 738.94 | 737.50 | 739.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 757.60 | 741.60 | 741.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 757.60 | 741.60 | 741.40 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 735.75 | 743.08 | 743.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 732.77 | 741.02 | 742.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 736.50 | 736.31 | 739.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 736.50 | 736.31 | 739.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 736.50 | 736.31 | 739.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:45:00 | 739.00 | 736.31 | 739.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 738.32 | 736.71 | 739.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 738.32 | 736.71 | 739.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 739.90 | 737.35 | 739.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 739.90 | 737.35 | 739.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 745.89 | 739.06 | 739.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 745.89 | 739.06 | 739.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 750.37 | 741.32 | 740.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 756.00 | 744.26 | 742.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 744.10 | 754.04 | 749.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 744.10 | 754.04 | 749.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 744.10 | 754.04 | 749.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 744.10 | 754.04 | 749.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 748.10 | 752.85 | 749.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 743.41 | 752.85 | 749.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 743.17 | 750.91 | 748.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 743.17 | 750.91 | 748.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 745.62 | 749.85 | 748.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:45:00 | 744.02 | 749.85 | 748.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 736.22 | 745.47 | 746.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 688.01 | 733.21 | 740.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 709.57 | 709.46 | 722.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 735.90 | 709.46 | 722.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 728.00 | 713.17 | 723.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 716.72 | 714.83 | 723.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 717.99 | 714.83 | 723.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:00:00 | 718.14 | 716.25 | 722.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:30:00 | 718.83 | 715.74 | 718.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 718.42 | 716.28 | 718.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 15:00:00 | 718.42 | 716.28 | 718.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 718.80 | 716.78 | 718.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:15:00 | 725.50 | 716.78 | 718.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 722.71 | 717.97 | 718.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 720.30 | 717.97 | 718.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 731.32 | 720.64 | 719.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 731.32 | 720.64 | 719.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 739.80 | 724.47 | 721.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 776.70 | 779.25 | 770.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 776.70 | 779.25 | 770.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 810.24 | 811.65 | 807.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 807.22 | 811.65 | 807.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 796.94 | 808.47 | 806.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 796.94 | 808.47 | 806.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 784.90 | 803.76 | 804.53 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 799.98 | 796.56 | 796.21 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 790.44 | 795.88 | 796.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 12:15:00 | 782.40 | 791.31 | 793.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 728.78 | 728.59 | 740.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 15:00:00 | 728.78 | 728.59 | 740.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 753.48 | 733.64 | 740.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 753.48 | 733.64 | 740.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 741.68 | 735.25 | 740.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 739.16 | 735.50 | 740.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 702.20 | 717.40 | 729.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 727.46 | 706.58 | 715.62 | SL hit (close>ema200) qty=0.50 sl=706.58 alert=retest2 |

### Cycle 68 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 731.92 | 721.79 | 720.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 735.72 | 726.53 | 723.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 722.76 | 726.81 | 724.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 13:15:00 | 722.76 | 726.81 | 724.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 722.76 | 726.81 | 724.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 722.76 | 726.81 | 724.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 724.16 | 726.28 | 724.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 15:15:00 | 727.20 | 726.28 | 724.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:15:00 | 727.02 | 726.34 | 725.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:00:00 | 728.00 | 726.97 | 725.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 773.60 | 776.75 | 777.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 10:15:00 | 773.60 | 776.75 | 777.17 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 787.00 | 778.68 | 777.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 795.34 | 787.51 | 782.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 792.20 | 794.13 | 788.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 14:00:00 | 792.20 | 794.13 | 788.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 795.70 | 796.87 | 794.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 794.22 | 796.87 | 794.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 791.06 | 795.71 | 794.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 791.06 | 795.71 | 794.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 794.40 | 795.45 | 794.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 797.48 | 795.45 | 794.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 800.16 | 797.98 | 796.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 847.30 | 853.61 | 853.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 847.30 | 853.61 | 853.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 833.60 | 848.62 | 851.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 807.12 | 799.91 | 811.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 807.12 | 799.91 | 811.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 812.20 | 802.37 | 811.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 812.20 | 802.37 | 811.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 824.00 | 806.70 | 812.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 824.00 | 806.70 | 812.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 822.02 | 809.76 | 813.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 823.70 | 809.76 | 813.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 843.74 | 816.56 | 815.96 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 816.82 | 825.14 | 825.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 810.72 | 822.25 | 823.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 817.06 | 812.23 | 817.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 817.06 | 812.23 | 817.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 817.06 | 812.23 | 817.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 815.48 | 812.23 | 817.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 828.60 | 815.50 | 818.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 828.60 | 815.50 | 818.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 836.50 | 819.70 | 820.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 836.08 | 819.70 | 820.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 831.60 | 822.08 | 821.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 841.80 | 830.35 | 825.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 845.90 | 846.65 | 840.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:30:00 | 846.74 | 846.65 | 840.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 845.04 | 848.69 | 844.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 845.38 | 848.69 | 844.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 844.98 | 847.95 | 844.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 839.14 | 847.95 | 844.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 840.00 | 846.36 | 844.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 840.30 | 846.36 | 844.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 837.70 | 844.63 | 843.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 838.64 | 844.63 | 843.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 833.24 | 842.35 | 842.88 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 860.20 | 846.16 | 844.40 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 847.18 | 853.49 | 853.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 15:15:00 | 845.60 | 848.46 | 850.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 852.40 | 849.25 | 850.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 852.40 | 849.25 | 850.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 852.40 | 849.25 | 850.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 852.40 | 849.25 | 850.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 853.88 | 850.17 | 850.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 853.88 | 850.17 | 850.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 851.80 | 850.50 | 850.87 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 852.38 | 851.17 | 851.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 854.20 | 851.90 | 851.49 | Break + close above crossover candle high |

### Cycle 79 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 841.04 | 849.75 | 850.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 834.60 | 842.24 | 846.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 826.78 | 825.97 | 831.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 826.78 | 825.97 | 831.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 826.78 | 825.97 | 831.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 829.42 | 825.97 | 831.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 837.78 | 828.33 | 831.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 837.78 | 828.33 | 831.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 837.60 | 830.19 | 832.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:30:00 | 834.80 | 831.15 | 832.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 834.40 | 832.44 | 833.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 838.20 | 834.50 | 834.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 15:15:00 | 838.20 | 834.50 | 834.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 844.18 | 836.43 | 834.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 13:15:00 | 836.98 | 837.25 | 835.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 13:15:00 | 836.98 | 837.25 | 835.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 836.98 | 837.25 | 835.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 837.28 | 837.25 | 835.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 836.00 | 837.12 | 836.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 817.40 | 837.12 | 836.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 815.34 | 832.76 | 834.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 812.84 | 828.78 | 832.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 822.58 | 819.42 | 824.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:45:00 | 822.40 | 819.42 | 824.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 822.20 | 819.98 | 824.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:30:00 | 818.24 | 821.62 | 824.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 829.90 | 823.28 | 825.44 | SL hit (close>static) qty=1.00 sl=826.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 835.82 | 826.75 | 826.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 842.04 | 829.81 | 827.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 840.80 | 843.33 | 837.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 11:00:00 | 840.80 | 843.33 | 837.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 842.40 | 843.75 | 839.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 849.64 | 843.75 | 839.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:30:00 | 845.98 | 846.51 | 843.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:30:00 | 842.44 | 844.52 | 843.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 843.96 | 844.52 | 843.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 844.60 | 844.54 | 844.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 843.64 | 844.54 | 844.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 845.00 | 844.63 | 844.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 841.82 | 844.63 | 844.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 848.46 | 845.40 | 844.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 850.20 | 846.73 | 845.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 842.26 | 846.11 | 845.42 | SL hit (close<static) qty=1.00 sl=844.14 alert=retest2 |

### Cycle 83 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 839.24 | 845.53 | 845.71 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 847.40 | 845.86 | 845.69 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 840.08 | 844.70 | 845.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 822.04 | 837.16 | 840.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 746.26 | 739.05 | 750.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:30:00 | 743.26 | 739.05 | 750.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 751.54 | 742.14 | 750.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 751.54 | 742.14 | 750.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 749.96 | 743.70 | 750.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 749.72 | 743.70 | 750.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 14:15:00 | 759.08 | 748.52 | 751.28 | SL hit (close>static) qty=1.00 sl=755.40 alert=retest2 |

### Cycle 86 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 758.24 | 752.66 | 752.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 11:15:00 | 764.16 | 758.57 | 756.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 756.28 | 760.35 | 758.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 756.28 | 760.35 | 758.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 756.28 | 760.35 | 758.28 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 751.26 | 757.25 | 757.26 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 764.86 | 758.57 | 757.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 766.76 | 760.21 | 758.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 10:15:00 | 758.22 | 760.01 | 758.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 10:15:00 | 758.22 | 760.01 | 758.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 758.22 | 760.01 | 758.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 758.22 | 760.01 | 758.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 759.58 | 759.93 | 758.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 758.20 | 759.93 | 758.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 756.26 | 759.19 | 758.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 756.26 | 759.19 | 758.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 754.08 | 758.17 | 758.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 748.66 | 756.27 | 757.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 13:15:00 | 755.46 | 750.89 | 753.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 13:15:00 | 755.46 | 750.89 | 753.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 755.46 | 750.89 | 753.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 755.46 | 750.89 | 753.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 760.36 | 752.78 | 754.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 759.68 | 752.78 | 754.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 760.60 | 754.35 | 754.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 757.78 | 754.35 | 754.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 760.32 | 755.54 | 755.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 765.92 | 758.00 | 756.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 763.98 | 764.73 | 761.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 09:15:00 | 776.96 | 765.31 | 761.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 776.00 | 772.36 | 767.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 771.18 | 772.36 | 767.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 773.28 | 777.04 | 773.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 774.26 | 777.04 | 773.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 774.40 | 776.51 | 773.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-19 12:15:00 | 773.00 | 775.44 | 773.33 | SL hit (close<ema400) qty=1.00 sl=773.33 alert=retest1 |

### Cycle 91 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 768.18 | 771.36 | 771.75 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 778.56 | 771.02 | 770.91 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 771.38 | 776.79 | 776.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 767.78 | 774.11 | 775.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 738.46 | 737.93 | 747.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:30:00 | 737.60 | 737.93 | 747.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 748.30 | 740.00 | 747.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 748.30 | 740.00 | 747.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 750.20 | 742.04 | 748.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 750.20 | 742.04 | 748.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 749.38 | 743.51 | 748.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:30:00 | 743.58 | 743.59 | 747.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 757.52 | 746.04 | 747.94 | SL hit (close>static) qty=1.00 sl=751.58 alert=retest2 |

### Cycle 94 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 760.64 | 750.82 | 749.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 764.20 | 753.50 | 751.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 759.42 | 762.66 | 758.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 759.42 | 762.66 | 758.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 759.42 | 762.66 | 758.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 759.42 | 762.66 | 758.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 760.66 | 762.26 | 758.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 760.66 | 762.26 | 758.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 762.00 | 762.21 | 759.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 760.20 | 762.21 | 759.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 764.40 | 762.65 | 759.65 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 757.06 | 759.37 | 759.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 752.20 | 757.94 | 759.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 761.60 | 758.23 | 758.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 761.60 | 758.23 | 758.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 761.60 | 758.23 | 758.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 761.60 | 758.23 | 758.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 763.62 | 759.31 | 759.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:30:00 | 763.02 | 759.31 | 759.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 764.30 | 760.31 | 759.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 766.46 | 762.26 | 760.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 769.34 | 770.57 | 768.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:15:00 | 774.60 | 770.57 | 768.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 776.90 | 778.67 | 775.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 776.90 | 778.67 | 775.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 770.00 | 776.93 | 774.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 770.00 | 776.93 | 774.97 | SL hit (close<ema400) qty=1.00 sl=774.97 alert=retest1 |

### Cycle 97 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 770.78 | 773.55 | 773.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 768.20 | 772.48 | 773.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 771.52 | 771.13 | 772.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 771.52 | 771.13 | 772.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 773.00 | 771.50 | 772.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 770.44 | 771.50 | 772.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 768.30 | 770.86 | 772.01 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 774.00 | 772.36 | 772.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 775.06 | 772.90 | 772.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 13:15:00 | 818.14 | 818.49 | 807.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 14:00:00 | 818.14 | 818.49 | 807.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 810.84 | 816.43 | 811.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 812.78 | 816.43 | 811.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 806.42 | 814.43 | 810.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:00:00 | 806.42 | 814.43 | 810.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 803.60 | 812.26 | 809.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 802.82 | 812.26 | 809.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 801.62 | 807.62 | 808.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 794.44 | 804.98 | 806.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 12:15:00 | 783.60 | 782.71 | 789.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:00:00 | 783.60 | 782.71 | 789.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 764.00 | 756.26 | 758.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 764.00 | 756.26 | 758.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 766.20 | 758.25 | 758.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 765.20 | 758.25 | 758.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 767.40 | 760.08 | 759.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 776.40 | 766.64 | 763.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 763.94 | 767.21 | 764.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 763.94 | 767.21 | 764.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 763.94 | 767.21 | 764.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 763.94 | 767.21 | 764.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 765.76 | 766.92 | 764.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 764.58 | 766.92 | 764.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 763.68 | 766.27 | 764.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 764.14 | 766.27 | 764.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 758.32 | 764.68 | 764.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 758.32 | 764.68 | 764.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 760.30 | 763.47 | 763.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 756.00 | 761.34 | 762.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 758.98 | 758.27 | 760.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:45:00 | 759.86 | 758.27 | 760.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 763.02 | 759.22 | 760.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 763.02 | 759.22 | 760.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 762.38 | 759.85 | 760.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:15:00 | 764.40 | 759.85 | 760.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 764.34 | 760.75 | 760.92 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 765.34 | 761.67 | 761.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 15:15:00 | 766.80 | 762.69 | 761.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 769.70 | 770.86 | 767.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 769.70 | 770.86 | 767.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 769.70 | 770.86 | 767.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 769.70 | 770.86 | 767.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 769.40 | 770.57 | 768.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:30:00 | 770.78 | 770.56 | 768.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 759.20 | 769.64 | 769.10 | SL hit (close<static) qty=1.00 sl=765.74 alert=retest2 |

### Cycle 103 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 759.26 | 767.56 | 768.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 755.72 | 765.19 | 767.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 769.46 | 763.73 | 765.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 769.46 | 763.73 | 765.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 769.46 | 763.73 | 765.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 769.46 | 763.73 | 765.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 771.20 | 765.23 | 766.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 771.20 | 765.23 | 766.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 771.64 | 766.89 | 766.83 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 764.40 | 766.82 | 767.12 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 769.52 | 767.36 | 767.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 11:15:00 | 771.80 | 768.25 | 767.74 | Break + close above crossover candle high |

### Cycle 107 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 761.80 | 767.15 | 767.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 758.96 | 765.51 | 766.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 764.18 | 762.64 | 764.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 13:15:00 | 764.18 | 762.64 | 764.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 764.18 | 762.64 | 764.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 764.08 | 762.64 | 764.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 766.14 | 763.34 | 764.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 765.80 | 763.34 | 764.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 765.90 | 763.85 | 764.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 767.40 | 764.79 | 764.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 766.60 | 765.15 | 765.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 774.00 | 766.92 | 765.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 14:15:00 | 776.00 | 776.22 | 773.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:45:00 | 774.16 | 776.22 | 773.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 768.20 | 790.09 | 788.02 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 764.22 | 784.92 | 785.86 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 12:15:00 | 790.78 | 783.19 | 782.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 792.00 | 786.15 | 784.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 787.90 | 787.97 | 785.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 787.90 | 787.97 | 785.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 787.90 | 787.97 | 785.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 787.90 | 787.97 | 785.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 787.82 | 789.21 | 787.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 787.82 | 789.21 | 787.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 784.06 | 788.18 | 786.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 778.40 | 788.18 | 786.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 777.60 | 786.07 | 785.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 779.28 | 786.07 | 785.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 783.72 | 785.60 | 785.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 769.04 | 778.82 | 781.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 750.78 | 747.13 | 755.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 750.78 | 747.13 | 755.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 751.80 | 748.06 | 755.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 755.00 | 748.06 | 755.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 759.36 | 750.32 | 755.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 761.40 | 750.32 | 755.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 760.20 | 752.30 | 755.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 764.26 | 752.30 | 755.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 760.60 | 757.95 | 757.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 767.60 | 759.88 | 758.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 759.20 | 760.63 | 759.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 759.20 | 760.63 | 759.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 759.20 | 760.63 | 759.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 759.26 | 760.63 | 759.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 758.70 | 760.24 | 759.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 756.80 | 760.24 | 759.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 761.52 | 760.50 | 759.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 764.80 | 760.50 | 759.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 784.02 | 788.12 | 788.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 784.02 | 788.12 | 788.50 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 806.32 | 791.12 | 789.60 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 789.40 | 794.79 | 794.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 787.64 | 793.36 | 794.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 788.96 | 781.04 | 784.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 788.96 | 781.04 | 784.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 788.96 | 781.04 | 784.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 779.82 | 784.00 | 784.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 789.70 | 781.25 | 781.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 789.70 | 781.25 | 781.01 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 770.88 | 780.22 | 780.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 766.06 | 776.23 | 778.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 773.60 | 771.87 | 774.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:00:00 | 773.60 | 771.87 | 774.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 772.74 | 772.05 | 774.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 771.00 | 772.05 | 774.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 780.20 | 773.67 | 774.53 | SL hit (close>static) qty=1.00 sl=774.44 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 782.20 | 775.38 | 775.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 11:15:00 | 790.40 | 778.38 | 776.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 766.00 | 782.23 | 780.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 766.00 | 782.23 | 780.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 766.00 | 782.23 | 780.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 766.00 | 782.23 | 780.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 766.20 | 779.02 | 778.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 760.10 | 779.02 | 778.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 766.50 | 776.52 | 777.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 758.90 | 769.97 | 773.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 746.10 | 742.32 | 748.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 746.10 | 742.32 | 748.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 744.00 | 742.65 | 748.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 749.90 | 742.65 | 748.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 752.10 | 744.54 | 748.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 752.10 | 744.54 | 748.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 750.90 | 745.81 | 748.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 754.10 | 745.81 | 748.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 757.00 | 750.70 | 750.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 761.80 | 752.92 | 751.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 15:15:00 | 754.20 | 755.14 | 753.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:15:00 | 750.50 | 755.14 | 753.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 751.80 | 754.47 | 753.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 749.00 | 754.47 | 753.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 753.00 | 754.18 | 753.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 750.90 | 754.18 | 753.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 753.70 | 754.08 | 753.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 752.10 | 754.08 | 753.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 756.80 | 754.63 | 753.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:30:00 | 754.30 | 754.63 | 753.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 754.90 | 755.04 | 754.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:30:00 | 754.40 | 755.04 | 754.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 754.30 | 754.90 | 754.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 749.40 | 754.90 | 754.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 748.10 | 753.54 | 753.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 748.50 | 753.54 | 753.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 752.30 | 753.29 | 753.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 746.50 | 750.97 | 752.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 741.30 | 739.98 | 744.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 741.30 | 739.98 | 744.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 745.50 | 741.09 | 744.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 743.50 | 741.09 | 744.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 746.90 | 742.25 | 744.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 746.90 | 742.25 | 744.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 752.20 | 744.24 | 745.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 752.20 | 744.24 | 745.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 753.10 | 746.01 | 746.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 754.90 | 746.01 | 746.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 753.80 | 747.57 | 746.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 759.60 | 753.78 | 750.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 760.90 | 761.44 | 757.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:30:00 | 761.30 | 761.44 | 757.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 759.70 | 761.09 | 757.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 758.00 | 761.09 | 757.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 760.50 | 760.83 | 758.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 759.50 | 760.83 | 758.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 771.10 | 762.83 | 759.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 774.50 | 762.83 | 759.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 755.00 | 761.08 | 760.67 | SL hit (close<static) qty=1.00 sl=757.20 alert=retest2 |

### Cycle 123 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 753.60 | 759.58 | 760.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 751.30 | 756.89 | 758.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 739.00 | 737.78 | 743.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 739.00 | 737.78 | 743.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 740.70 | 737.36 | 741.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 740.70 | 737.36 | 741.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 742.10 | 738.66 | 741.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 742.10 | 738.66 | 741.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 744.30 | 739.79 | 741.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 743.60 | 739.79 | 741.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 741.10 | 740.05 | 741.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 737.75 | 739.72 | 741.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 740.35 | 740.23 | 741.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 747.50 | 739.83 | 740.39 | SL hit (close>static) qty=1.00 sl=744.50 alert=retest2 |

### Cycle 124 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 751.90 | 742.25 | 741.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 755.00 | 744.80 | 742.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 747.05 | 751.72 | 748.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 747.05 | 751.72 | 748.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 747.05 | 751.72 | 748.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 747.05 | 751.72 | 748.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 753.40 | 752.06 | 749.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 761.80 | 750.21 | 749.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 13:30:00 | 754.55 | 752.99 | 751.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 754.50 | 753.79 | 751.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 755.65 | 754.65 | 752.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 750.00 | 753.72 | 752.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 750.00 | 753.72 | 752.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 751.65 | 753.31 | 752.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 751.65 | 753.31 | 752.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 744.35 | 750.90 | 751.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 744.35 | 750.90 | 751.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 740.50 | 747.46 | 749.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 710.70 | 707.67 | 716.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 710.70 | 707.67 | 716.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 727.65 | 712.49 | 717.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 728.05 | 712.49 | 717.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 726.75 | 715.34 | 718.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 726.75 | 715.34 | 718.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 722.00 | 719.62 | 719.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 723.50 | 719.62 | 719.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 721.95 | 720.09 | 719.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 729.40 | 721.95 | 720.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 721.40 | 721.84 | 720.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 10:15:00 | 721.40 | 721.84 | 720.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 721.40 | 721.84 | 720.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 721.40 | 721.84 | 720.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 731.00 | 723.67 | 721.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 734.05 | 724.97 | 722.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 717.90 | 723.35 | 723.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 717.90 | 723.35 | 723.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 712.70 | 719.86 | 721.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 702.45 | 701.85 | 708.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 710.45 | 701.85 | 708.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 713.90 | 704.26 | 708.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 712.95 | 704.26 | 708.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 709.50 | 705.30 | 708.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 708.65 | 705.30 | 708.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 12:15:00 | 727.30 | 709.70 | 710.29 | SL hit (close>static) qty=1.00 sl=715.50 alert=retest2 |

### Cycle 128 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 707.65 | 702.13 | 701.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 708.75 | 704.63 | 702.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 703.95 | 705.92 | 704.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 703.95 | 705.92 | 704.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 703.95 | 705.92 | 704.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 703.90 | 705.92 | 704.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 703.40 | 705.42 | 704.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 703.40 | 705.42 | 704.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 700.80 | 704.50 | 703.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:45:00 | 701.00 | 704.50 | 703.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 703.55 | 704.31 | 703.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:30:00 | 709.00 | 704.86 | 704.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 705.10 | 705.08 | 704.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 699.20 | 703.26 | 703.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 699.20 | 703.26 | 703.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 695.95 | 701.80 | 702.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 699.65 | 698.90 | 700.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 699.65 | 698.90 | 700.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 699.65 | 698.90 | 700.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 698.80 | 698.90 | 700.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 700.80 | 699.28 | 700.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 700.80 | 699.28 | 700.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 690.25 | 697.47 | 699.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:30:00 | 697.05 | 697.47 | 699.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 693.85 | 684.52 | 688.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 693.85 | 684.52 | 688.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 693.00 | 686.22 | 689.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 723.50 | 686.22 | 689.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 721.95 | 693.36 | 692.13 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 715.25 | 720.96 | 721.10 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 723.50 | 721.18 | 721.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 729.95 | 723.13 | 722.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 740.10 | 740.64 | 735.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 740.10 | 740.64 | 735.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 735.05 | 742.99 | 740.02 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 733.20 | 738.11 | 738.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 715.95 | 730.88 | 734.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 730.65 | 725.65 | 729.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 730.65 | 725.65 | 729.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 730.65 | 725.65 | 729.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 730.65 | 725.65 | 729.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 732.55 | 727.03 | 729.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 732.55 | 727.03 | 729.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 730.60 | 727.75 | 729.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:45:00 | 729.30 | 728.33 | 729.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 737.05 | 730.93 | 730.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 737.05 | 730.93 | 730.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 743.60 | 734.60 | 732.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 732.60 | 734.20 | 732.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 10:15:00 | 732.60 | 734.20 | 732.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 732.60 | 734.20 | 732.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 732.60 | 734.20 | 732.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 732.45 | 733.85 | 732.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 732.45 | 733.85 | 732.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 734.20 | 733.92 | 732.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:30:00 | 736.75 | 733.93 | 732.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 738.00 | 734.18 | 733.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 736.65 | 734.71 | 733.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 730.00 | 733.30 | 733.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 730.00 | 733.30 | 733.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 728.15 | 731.82 | 732.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 727.40 | 723.02 | 725.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 727.40 | 723.02 | 725.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 727.40 | 723.02 | 725.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 728.15 | 723.02 | 725.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 721.25 | 722.67 | 725.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 718.00 | 721.91 | 724.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 717.45 | 721.19 | 724.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 711.95 | 719.17 | 722.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 682.10 | 699.72 | 706.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 681.58 | 699.72 | 706.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 15:15:00 | 676.35 | 686.20 | 696.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 12:15:00 | 646.20 | 665.93 | 682.69 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 136 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 659.00 | 649.45 | 648.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 676.50 | 666.53 | 659.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 657.85 | 670.90 | 666.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 657.85 | 670.90 | 666.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 657.85 | 670.90 | 666.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 657.85 | 670.90 | 666.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 661.80 | 669.08 | 665.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:30:00 | 657.00 | 669.08 | 665.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 664.20 | 666.19 | 665.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 663.10 | 666.19 | 665.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 663.75 | 665.70 | 665.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 655.75 | 665.70 | 665.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 652.40 | 663.04 | 663.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 648.45 | 657.44 | 660.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 639.20 | 636.66 | 645.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 639.20 | 636.66 | 645.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 639.20 | 636.66 | 645.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 643.50 | 636.66 | 645.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 635.80 | 636.65 | 644.22 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 655.00 | 643.49 | 642.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 658.80 | 648.58 | 645.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 645.00 | 649.37 | 646.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 645.00 | 649.37 | 646.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 645.00 | 649.37 | 646.30 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 638.50 | 644.95 | 645.16 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 647.05 | 645.44 | 645.31 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 642.75 | 644.91 | 645.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 640.50 | 643.32 | 644.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 630.25 | 625.94 | 631.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 630.25 | 625.94 | 631.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 630.25 | 625.94 | 631.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 627.80 | 625.94 | 631.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 653.00 | 637.19 | 635.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 653.00 | 637.19 | 635.10 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 624.50 | 638.45 | 639.71 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 666.75 | 637.43 | 637.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 669.25 | 659.98 | 653.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 698.90 | 702.02 | 691.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 699.25 | 702.02 | 691.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 703.65 | 710.80 | 703.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 707.45 | 710.80 | 703.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 709.20 | 710.67 | 704.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 706.85 | 708.36 | 704.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:15:00 | 706.75 | 708.36 | 704.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 706.15 | 707.92 | 704.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 706.15 | 707.92 | 704.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 706.00 | 707.53 | 704.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 715.05 | 707.53 | 704.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 10:15:00 | 777.54 | 760.97 | 754.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 754.50 | 760.59 | 761.00 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 767.50 | 761.67 | 761.08 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 737.20 | 756.74 | 759.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 733.20 | 752.03 | 756.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 744.00 | 743.82 | 749.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 744.00 | 743.82 | 749.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 744.00 | 743.82 | 749.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 736.00 | 741.28 | 747.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 779.45 | 743.37 | 745.15 | SL hit (close>static) qty=1.00 sl=750.60 alert=retest2 |

### Cycle 148 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 773.00 | 749.30 | 747.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 793.85 | 758.21 | 751.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 835.00 | 835.01 | 822.66 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-30 10:30:00 | 678.51 | 2024-05-30 15:15:00 | 696.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest1 | 2024-06-11 09:15:00 | 697.00 | 2024-06-11 14:15:00 | 687.93 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-06-12 09:15:00 | 699.72 | 2024-06-20 09:15:00 | 709.98 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2024-07-05 11:15:00 | 748.69 | 2024-07-10 09:15:00 | 823.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-08 12:15:00 | 751.01 | 2024-07-10 09:15:00 | 826.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-24 14:00:00 | 810.96 | 2024-07-29 09:15:00 | 845.76 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2024-07-25 09:15:00 | 808.85 | 2024-07-29 09:15:00 | 845.76 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2024-07-26 09:45:00 | 812.00 | 2024-07-29 09:15:00 | 845.76 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2024-08-07 12:30:00 | 816.74 | 2024-08-08 09:15:00 | 862.68 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest2 | 2024-08-12 12:45:00 | 854.92 | 2024-08-13 10:15:00 | 848.74 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-08-12 13:15:00 | 855.60 | 2024-08-13 10:15:00 | 848.74 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-08-13 09:15:00 | 857.98 | 2024-08-13 10:15:00 | 848.74 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-08-21 09:15:00 | 887.08 | 2024-08-22 14:15:00 | 876.39 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-08-21 09:45:00 | 893.59 | 2024-08-22 14:15:00 | 876.39 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-08-22 12:30:00 | 885.88 | 2024-08-22 14:15:00 | 876.39 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-08-29 10:30:00 | 869.06 | 2024-09-02 14:15:00 | 883.19 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-08-30 09:30:00 | 864.55 | 2024-09-02 14:15:00 | 883.19 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-08-30 11:00:00 | 868.62 | 2024-09-02 14:15:00 | 883.19 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-09-02 11:30:00 | 866.10 | 2024-09-02 14:15:00 | 883.19 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-09-04 09:15:00 | 882.62 | 2024-09-06 10:15:00 | 871.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-09-13 15:00:00 | 880.41 | 2024-09-19 12:15:00 | 876.38 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-09-19 12:00:00 | 880.40 | 2024-09-19 12:15:00 | 876.38 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-10-15 09:30:00 | 936.59 | 2024-10-21 10:15:00 | 942.64 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2024-10-16 09:15:00 | 947.60 | 2024-10-21 10:15:00 | 942.64 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-10-18 09:30:00 | 959.68 | 2024-10-21 10:15:00 | 942.64 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-10-24 09:30:00 | 904.07 | 2024-10-25 10:15:00 | 858.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 10:00:00 | 908.40 | 2024-10-25 10:15:00 | 862.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 10:45:00 | 907.01 | 2024-10-25 10:15:00 | 861.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 09:30:00 | 904.07 | 2024-10-29 11:15:00 | 860.20 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2024-10-24 10:00:00 | 908.40 | 2024-10-29 11:15:00 | 860.20 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2024-10-24 10:45:00 | 907.01 | 2024-10-29 11:15:00 | 860.20 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest1 | 2024-11-05 09:15:00 | 913.53 | 2024-11-12 09:15:00 | 959.21 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-11-05 09:15:00 | 913.53 | 2024-11-12 13:15:00 | 948.84 | STOP_HIT | 0.50 | 3.87% |
| BUY | retest2 | 2024-11-05 13:45:00 | 915.00 | 2024-11-13 09:15:00 | 904.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-11-05 14:15:00 | 914.13 | 2024-11-13 09:15:00 | 904.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-11-05 15:00:00 | 914.06 | 2024-11-13 09:15:00 | 904.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-11-06 09:15:00 | 921.77 | 2024-11-13 09:15:00 | 904.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-11-12 09:15:00 | 960.60 | 2024-11-13 09:15:00 | 904.00 | STOP_HIT | 1.00 | -5.89% |
| SELL | retest2 | 2024-11-18 09:30:00 | 892.40 | 2024-11-25 14:15:00 | 907.58 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-11-18 12:30:00 | 899.98 | 2024-11-25 14:15:00 | 907.58 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-11-18 13:45:00 | 900.42 | 2024-11-25 14:15:00 | 907.58 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-11-21 09:15:00 | 899.11 | 2024-11-25 14:15:00 | 907.58 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-11-22 09:45:00 | 886.70 | 2024-11-25 14:15:00 | 907.58 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-12-16 14:15:00 | 1038.03 | 2024-12-23 09:15:00 | 986.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:45:00 | 1036.20 | 2024-12-23 09:15:00 | 984.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 14:15:00 | 1038.03 | 2024-12-24 12:15:00 | 983.44 | STOP_HIT | 0.50 | 5.26% |
| SELL | retest2 | 2024-12-17 09:45:00 | 1036.20 | 2024-12-24 12:15:00 | 983.44 | STOP_HIT | 0.50 | 5.09% |
| BUY | retest2 | 2024-12-30 15:15:00 | 1020.60 | 2024-12-31 09:15:00 | 975.75 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2024-12-31 14:00:00 | 1019.08 | 2025-01-01 11:15:00 | 997.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-01-01 15:00:00 | 1021.87 | 2025-01-06 10:15:00 | 985.21 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-01-02 11:45:00 | 1020.49 | 2025-01-06 10:15:00 | 985.21 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-01-15 12:15:00 | 858.86 | 2025-01-16 09:15:00 | 901.32 | STOP_HIT | 1.00 | -4.94% |
| SELL | retest2 | 2025-01-15 13:00:00 | 860.45 | 2025-01-16 09:15:00 | 901.32 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2025-01-20 13:00:00 | 888.44 | 2025-01-21 09:15:00 | 865.98 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-01-20 14:30:00 | 887.80 | 2025-01-21 09:15:00 | 865.98 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-01-21 09:15:00 | 891.46 | 2025-01-21 09:15:00 | 865.98 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-01-24 14:30:00 | 817.41 | 2025-01-27 09:15:00 | 776.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:30:00 | 817.41 | 2025-01-28 10:15:00 | 735.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 687.07 | 2025-02-14 13:15:00 | 662.62 | PARTIAL | 0.50 | 3.56% |
| SELL | retest2 | 2025-02-13 09:15:00 | 687.07 | 2025-02-14 15:15:00 | 678.41 | STOP_HIT | 0.50 | 1.26% |
| SELL | retest2 | 2025-02-13 10:15:00 | 697.50 | 2025-02-17 09:15:00 | 660.78 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2025-02-13 11:30:00 | 695.56 | 2025-02-17 09:15:00 | 659.41 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2025-02-13 10:15:00 | 697.50 | 2025-02-17 13:15:00 | 671.42 | STOP_HIT | 0.50 | 3.74% |
| SELL | retest2 | 2025-02-13 11:30:00 | 695.56 | 2025-02-17 13:15:00 | 671.42 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2025-02-13 12:00:00 | 694.12 | 2025-02-18 13:15:00 | 652.72 | PARTIAL | 0.50 | 5.96% |
| SELL | retest2 | 2025-02-13 12:00:00 | 694.12 | 2025-02-19 09:15:00 | 677.19 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2025-02-17 09:30:00 | 667.26 | 2025-02-19 12:15:00 | 682.93 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-02-17 13:45:00 | 669.25 | 2025-02-19 12:15:00 | 682.93 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-02-18 09:15:00 | 666.73 | 2025-02-19 12:15:00 | 682.93 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-02-19 10:45:00 | 670.89 | 2025-02-19 12:15:00 | 682.93 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-02-24 11:30:00 | 684.27 | 2025-02-24 13:15:00 | 681.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-02-24 12:00:00 | 680.38 | 2025-02-24 13:15:00 | 681.90 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-03-12 09:30:00 | 680.10 | 2025-03-12 14:15:00 | 664.75 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-03-12 10:15:00 | 676.21 | 2025-03-12 14:15:00 | 664.75 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-03-26 14:45:00 | 737.48 | 2025-03-28 09:15:00 | 757.60 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-03-26 15:15:00 | 738.00 | 2025-03-28 09:15:00 | 757.60 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-03-27 10:15:00 | 738.72 | 2025-03-28 09:15:00 | 757.60 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-03-27 10:45:00 | 738.41 | 2025-03-28 09:15:00 | 757.60 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-04-08 10:30:00 | 716.72 | 2025-04-11 10:15:00 | 731.32 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-04-08 11:15:00 | 717.99 | 2025-04-11 10:15:00 | 731.32 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-04-08 13:00:00 | 718.14 | 2025-04-11 10:15:00 | 731.32 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-04-09 13:30:00 | 718.83 | 2025-04-11 10:15:00 | 731.32 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-04-11 10:15:00 | 720.30 | 2025-04-11 10:15:00 | 731.32 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-08 11:30:00 | 739.16 | 2025-05-09 09:15:00 | 702.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 11:30:00 | 739.16 | 2025-05-12 09:15:00 | 727.46 | STOP_HIT | 0.50 | 1.58% |
| BUY | retest2 | 2025-05-13 15:15:00 | 727.20 | 2025-05-23 10:15:00 | 773.60 | STOP_HIT | 1.00 | 6.38% |
| BUY | retest2 | 2025-05-14 10:15:00 | 727.02 | 2025-05-23 10:15:00 | 773.60 | STOP_HIT | 1.00 | 6.41% |
| BUY | retest2 | 2025-05-14 13:00:00 | 728.00 | 2025-05-23 10:15:00 | 773.60 | STOP_HIT | 1.00 | 6.26% |
| BUY | retest2 | 2025-05-29 15:15:00 | 797.48 | 2025-06-11 14:15:00 | 847.30 | STOP_HIT | 1.00 | 6.25% |
| BUY | retest2 | 2025-06-02 09:30:00 | 800.16 | 2025-06-11 14:15:00 | 847.30 | STOP_HIT | 1.00 | 5.89% |
| SELL | retest2 | 2025-07-09 12:30:00 | 834.80 | 2025-07-09 15:15:00 | 838.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-09 14:15:00 | 834.40 | 2025-07-09 15:15:00 | 838.20 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-14 11:30:00 | 818.24 | 2025-07-14 12:15:00 | 829.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-17 09:15:00 | 849.64 | 2025-07-22 10:15:00 | 842.26 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-18 10:30:00 | 845.98 | 2025-07-23 09:15:00 | 843.68 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-07-21 09:30:00 | 842.44 | 2025-07-23 10:15:00 | 839.24 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-07-21 10:00:00 | 843.96 | 2025-07-23 10:15:00 | 839.24 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-21 15:15:00 | 850.20 | 2025-07-23 10:15:00 | 839.24 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-07-22 15:00:00 | 850.52 | 2025-07-23 10:15:00 | 839.24 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-08-04 13:15:00 | 749.72 | 2025-08-04 14:15:00 | 759.08 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest1 | 2025-08-14 09:15:00 | 776.96 | 2025-08-19 12:15:00 | 773.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-08-29 14:30:00 | 743.58 | 2025-09-01 09:15:00 | 757.52 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest1 | 2025-09-10 09:15:00 | 774.60 | 2025-09-11 13:15:00 | 770.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-13 12:30:00 | 770.78 | 2025-10-14 11:15:00 | 759.20 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-11-11 12:15:00 | 764.80 | 2025-11-19 10:15:00 | 784.02 | STOP_HIT | 1.00 | 2.51% |
| SELL | retest2 | 2025-11-27 13:15:00 | 779.82 | 2025-12-01 10:15:00 | 789.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-12-03 14:15:00 | 771.00 | 2025-12-04 09:15:00 | 780.20 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-24 10:15:00 | 774.50 | 2025-12-26 09:15:00 | 755.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-01-01 09:30:00 | 737.75 | 2026-01-02 09:15:00 | 747.50 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-01 12:15:00 | 740.35 | 2026-01-02 09:15:00 | 747.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-07 09:15:00 | 761.80 | 2026-01-08 13:15:00 | 744.35 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-01-07 13:30:00 | 754.55 | 2026-01-08 13:15:00 | 744.35 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-01-07 14:30:00 | 754.50 | 2026-01-08 13:15:00 | 744.35 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-01-08 10:00:00 | 755.65 | 2026-01-08 13:15:00 | 744.35 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-16 12:30:00 | 734.05 | 2026-01-20 09:15:00 | 717.90 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-01-22 11:15:00 | 708.65 | 2026-01-22 12:15:00 | 727.30 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-01-22 15:00:00 | 706.55 | 2026-01-28 11:15:00 | 707.65 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-01-23 10:15:00 | 708.50 | 2026-01-28 11:15:00 | 707.65 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-01-27 10:00:00 | 708.25 | 2026-01-28 11:15:00 | 707.65 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2026-01-27 12:00:00 | 699.90 | 2026-01-28 11:15:00 | 707.65 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-01-27 13:00:00 | 698.95 | 2026-01-28 11:15:00 | 707.65 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-01-29 14:30:00 | 709.00 | 2026-01-30 11:15:00 | 699.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-01-30 09:30:00 | 705.10 | 2026-01-30 11:15:00 | 699.20 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-16 12:45:00 | 729.30 | 2026-02-16 14:15:00 | 737.05 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-02-17 13:30:00 | 736.75 | 2026-02-19 11:15:00 | 730.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-18 09:15:00 | 738.00 | 2026-02-19 11:15:00 | 730.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-02-18 14:45:00 | 736.65 | 2026-02-19 11:15:00 | 730.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-02-23 11:45:00 | 718.00 | 2026-02-27 10:15:00 | 682.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:15:00 | 717.45 | 2026-02-27 10:15:00 | 681.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 711.95 | 2026-02-27 15:15:00 | 676.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:45:00 | 718.00 | 2026-03-02 12:15:00 | 646.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 13:15:00 | 717.45 | 2026-03-02 12:15:00 | 645.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 711.95 | 2026-03-04 09:15:00 | 640.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 10:15:00 | 627.80 | 2026-03-25 09:15:00 | 653.00 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2026-04-13 10:15:00 | 707.45 | 2026-04-23 10:15:00 | 777.54 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2026-04-13 10:45:00 | 709.20 | 2026-04-23 10:15:00 | 777.43 | TARGET_HIT | 1.00 | 9.62% |
| BUY | retest2 | 2026-04-13 13:45:00 | 706.85 | 2026-04-28 09:15:00 | 754.50 | STOP_HIT | 1.00 | 6.74% |
| BUY | retest2 | 2026-04-13 14:15:00 | 706.75 | 2026-04-28 09:15:00 | 754.50 | STOP_HIT | 1.00 | 6.76% |
| BUY | retest2 | 2026-04-15 09:15:00 | 715.05 | 2026-04-28 09:15:00 | 754.50 | STOP_HIT | 1.00 | 5.52% |
| SELL | retest2 | 2026-05-04 11:45:00 | 736.00 | 2026-05-05 09:15:00 | 779.45 | STOP_HIT | 1.00 | -5.90% |
