# Lloyds Metals And Energy Ltd. (LLOYDSME)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-11 15:15:00 (3717 bars)
- **Last close:** 1755.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 162 |
| ALERT1 | 103 |
| ALERT2 | 101 |
| ALERT2_SKIP | 47 |
| ALERT3 | 293 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 129 |
| PARTIAL | 21 |
| TARGET_HIT | 12 |
| STOP_HIT | 122 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 155 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 67 / 88
- **Target hits / Stop hits / Partials:** 12 / 122 / 21
- **Avg / median % per leg:** 1.18% / -0.66%
- **Sum % (uncompounded):** 183.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 22 | 29.7% | 11 | 63 | 0 | 0.59% | 43.5% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.67% | -6.7% |
| BUY @ 3rd Alert (retest2) | 70 | 22 | 31.4% | 11 | 59 | 0 | 0.72% | 50.1% |
| SELL (all) | 81 | 45 | 55.6% | 1 | 59 | 21 | 1.73% | 140.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.57% | -3.6% |
| SELL @ 3rd Alert (retest2) | 80 | 45 | 56.2% | 1 | 58 | 21 | 1.80% | 143.6% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.05% | -10.2% |
| retest2 (combined) | 150 | 67 | 44.7% | 12 | 117 | 21 | 1.29% | 193.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 685.35 | 678.83 | 678.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 691.00 | 681.26 | 679.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 15:15:00 | 691.20 | 697.54 | 691.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 15:15:00 | 691.20 | 697.54 | 691.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 691.20 | 697.54 | 691.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 09:15:00 | 705.25 | 697.54 | 691.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 09:15:00 | 690.20 | 694.47 | 693.34 | SL hit (close<static) qty=1.00 sl=691.05 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 10:15:00 | 685.00 | 692.57 | 692.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 11:15:00 | 682.10 | 690.48 | 691.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 09:15:00 | 710.00 | 690.30 | 690.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 710.00 | 690.30 | 690.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 710.00 | 690.30 | 690.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:00:00 | 710.00 | 690.30 | 690.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 723.25 | 696.89 | 693.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 10:15:00 | 725.00 | 713.47 | 706.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 715.95 | 717.02 | 710.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 715.95 | 717.02 | 710.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 716.50 | 716.59 | 711.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 714.60 | 716.59 | 711.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 708.50 | 714.97 | 711.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:45:00 | 708.75 | 714.97 | 711.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 708.00 | 713.58 | 710.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 708.00 | 713.58 | 710.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 706.00 | 709.20 | 709.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 09:15:00 | 702.35 | 707.83 | 708.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 707.70 | 705.62 | 707.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 13:15:00 | 707.70 | 705.62 | 707.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 707.70 | 705.62 | 707.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 707.70 | 705.62 | 707.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 705.60 | 705.61 | 707.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:45:00 | 707.20 | 705.61 | 707.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 708.00 | 706.09 | 707.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 709.10 | 706.64 | 707.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 701.90 | 705.69 | 706.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:45:00 | 700.00 | 703.49 | 705.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 10:45:00 | 700.55 | 702.80 | 704.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 11:15:00 | 693.55 | 687.03 | 686.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 693.55 | 687.03 | 686.70 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 672.50 | 686.93 | 687.17 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 706.50 | 662.16 | 659.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 15:15:00 | 709.20 | 698.15 | 687.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 11:15:00 | 718.85 | 721.01 | 710.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 12:00:00 | 718.85 | 721.01 | 710.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 719.65 | 724.59 | 720.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 720.75 | 724.59 | 720.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 720.20 | 723.72 | 720.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:30:00 | 720.60 | 723.72 | 720.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 719.65 | 722.90 | 720.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:45:00 | 723.00 | 721.73 | 720.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 15:15:00 | 718.90 | 721.17 | 720.31 | SL hit (close<static) qty=1.00 sl=719.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 718.00 | 731.22 | 731.88 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 735.45 | 731.32 | 731.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 15:15:00 | 738.00 | 734.38 | 732.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 11:15:00 | 734.10 | 736.45 | 734.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 11:15:00 | 734.10 | 736.45 | 734.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 734.10 | 736.45 | 734.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:30:00 | 732.95 | 736.45 | 734.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 732.90 | 735.74 | 734.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 732.90 | 735.74 | 734.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 734.95 | 735.58 | 734.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:15:00 | 730.00 | 735.58 | 734.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 735.00 | 735.47 | 734.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:30:00 | 732.25 | 735.47 | 734.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 732.95 | 734.96 | 734.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 731.75 | 734.96 | 734.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 730.00 | 733.97 | 733.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 731.70 | 733.97 | 733.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 739.90 | 735.16 | 734.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 729.60 | 735.16 | 734.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 736.00 | 735.32 | 734.55 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 729.20 | 733.94 | 734.11 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 10:15:00 | 739.65 | 734.30 | 734.13 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 729.30 | 733.14 | 733.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 726.20 | 731.75 | 732.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 736.00 | 731.50 | 732.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 736.00 | 731.50 | 732.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 736.00 | 731.50 | 732.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:30:00 | 737.60 | 731.50 | 732.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 734.35 | 732.07 | 732.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:30:00 | 736.45 | 732.07 | 732.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 733.55 | 732.37 | 732.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 735.65 | 732.37 | 732.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 737.10 | 733.32 | 733.20 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 13:15:00 | 732.10 | 733.07 | 733.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 727.20 | 731.90 | 732.57 | Break + close below crossover candle low |

### Cycle 15 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 740.00 | 733.12 | 732.98 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 728.75 | 733.06 | 733.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 14:15:00 | 726.50 | 731.75 | 732.57 | Break + close below crossover candle low |

### Cycle 17 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 739.30 | 733.18 | 733.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 747.15 | 736.90 | 735.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 11:15:00 | 737.20 | 737.61 | 735.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 11:45:00 | 738.85 | 737.61 | 735.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 739.55 | 738.00 | 736.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 09:15:00 | 743.00 | 736.41 | 735.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 11:15:00 | 730.85 | 735.55 | 735.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 730.85 | 735.55 | 735.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 728.75 | 734.19 | 735.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 733.85 | 732.69 | 733.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 733.85 | 732.69 | 733.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 733.85 | 732.69 | 733.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 11:15:00 | 730.60 | 732.77 | 733.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 13:15:00 | 741.50 | 735.76 | 735.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 13:15:00 | 741.50 | 735.76 | 735.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 779.00 | 745.11 | 739.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 10:15:00 | 768.05 | 768.19 | 757.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 11:00:00 | 768.05 | 768.19 | 757.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 760.80 | 765.99 | 759.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:45:00 | 760.35 | 765.99 | 759.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 762.25 | 765.24 | 759.45 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 753.35 | 757.59 | 757.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 12:15:00 | 751.55 | 755.41 | 756.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 755.60 | 755.45 | 756.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 755.60 | 755.45 | 756.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 755.60 | 755.45 | 756.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:45:00 | 759.00 | 755.45 | 756.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 761.00 | 756.56 | 756.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:45:00 | 763.50 | 756.56 | 756.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 15:15:00 | 760.00 | 757.25 | 757.22 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 747.85 | 755.37 | 756.36 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 761.45 | 756.78 | 756.75 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 756.00 | 756.62 | 756.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 14:15:00 | 754.25 | 756.15 | 756.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 10:15:00 | 755.05 | 754.63 | 755.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 10:15:00 | 755.05 | 754.63 | 755.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 755.05 | 754.63 | 755.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 755.05 | 754.63 | 755.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 749.20 | 753.54 | 755.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:15:00 | 748.60 | 753.54 | 755.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 13:00:00 | 748.60 | 752.56 | 754.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 757.25 | 750.30 | 751.91 | SL hit (close>static) qty=1.00 sl=756.50 alert=retest2 |

### Cycle 25 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 717.05 | 712.99 | 712.84 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 14:15:00 | 710.35 | 712.73 | 712.77 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 09:15:00 | 715.70 | 712.79 | 712.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 11:15:00 | 720.85 | 715.24 | 713.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 735.15 | 740.55 | 733.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:00:00 | 735.15 | 740.55 | 733.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 734.45 | 739.33 | 733.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 13:45:00 | 734.35 | 739.33 | 733.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 733.85 | 738.23 | 733.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 15:15:00 | 738.30 | 738.23 | 733.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:00:00 | 738.95 | 738.39 | 734.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:15:00 | 745.00 | 737.07 | 734.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 15:15:00 | 740.00 | 739.12 | 736.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 740.00 | 739.30 | 736.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 749.65 | 739.30 | 736.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:15:00 | 742.45 | 739.83 | 737.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:00:00 | 744.75 | 740.81 | 737.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 14:15:00 | 733.15 | 737.29 | 736.95 | SL hit (close<static) qty=1.00 sl=734.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 729.00 | 735.63 | 736.23 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 785.05 | 745.51 | 740.67 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 731.65 | 759.11 | 762.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 714.45 | 731.30 | 742.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 727.05 | 725.74 | 736.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 09:30:00 | 724.70 | 725.74 | 736.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 736.70 | 727.93 | 736.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 737.50 | 727.93 | 736.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 731.85 | 728.72 | 736.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 736.50 | 728.72 | 736.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 736.65 | 730.30 | 736.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:00:00 | 730.15 | 734.15 | 736.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 739.50 | 735.22 | 736.90 | SL hit (close>static) qty=1.00 sl=739.10 alert=retest2 |

### Cycle 31 — BUY (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 13:15:00 | 744.20 | 738.20 | 737.91 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 736.00 | 737.41 | 737.58 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 742.90 | 738.51 | 738.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 10:15:00 | 749.90 | 740.78 | 739.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 14:15:00 | 759.05 | 760.37 | 754.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 15:00:00 | 759.05 | 760.37 | 754.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 759.30 | 759.93 | 755.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:00:00 | 759.30 | 759.93 | 755.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 755.65 | 758.45 | 755.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 756.95 | 758.45 | 755.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 757.45 | 758.25 | 755.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 757.45 | 758.25 | 755.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 745.15 | 755.63 | 754.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 745.15 | 755.63 | 754.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 745.80 | 753.66 | 753.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 10:15:00 | 731.05 | 747.42 | 750.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 739.55 | 736.17 | 740.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 13:30:00 | 738.50 | 736.17 | 740.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 739.90 | 736.92 | 740.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:30:00 | 741.95 | 736.92 | 740.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 739.00 | 737.33 | 740.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 748.30 | 737.33 | 740.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 740.80 | 738.03 | 740.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:45:00 | 736.30 | 737.47 | 740.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 11:30:00 | 734.70 | 736.70 | 739.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 15:15:00 | 752.65 | 742.99 | 741.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 15:15:00 | 752.65 | 742.99 | 741.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 757.00 | 749.16 | 746.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 11:15:00 | 770.50 | 774.04 | 766.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 11:45:00 | 772.35 | 774.04 | 766.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 770.45 | 773.32 | 767.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 15:00:00 | 774.00 | 772.78 | 767.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 772.65 | 772.20 | 768.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:00:00 | 774.75 | 772.27 | 768.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 15:00:00 | 779.85 | 780.79 | 777.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 781.00 | 780.83 | 777.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 784.05 | 780.83 | 777.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:45:00 | 784.25 | 781.40 | 778.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 770.85 | 779.85 | 779.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 770.85 | 779.85 | 779.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 765.10 | 776.90 | 778.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 774.10 | 770.51 | 774.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 774.10 | 770.51 | 774.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 774.10 | 770.51 | 774.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 751.75 | 772.99 | 773.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 764.50 | 765.06 | 767.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 11:00:00 | 765.60 | 765.17 | 767.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 13:00:00 | 766.55 | 765.07 | 766.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 768.50 | 765.75 | 766.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:30:00 | 769.05 | 765.75 | 766.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 771.95 | 766.99 | 767.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 771.95 | 766.99 | 767.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 772.85 | 768.23 | 767.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 772.85 | 768.23 | 767.92 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 763.70 | 767.29 | 767.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 12:15:00 | 762.65 | 766.36 | 767.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 13:15:00 | 770.85 | 767.26 | 767.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 13:15:00 | 770.85 | 767.26 | 767.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 770.85 | 767.26 | 767.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:00:00 | 770.85 | 767.26 | 767.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 778.00 | 769.41 | 768.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 11:15:00 | 783.50 | 776.42 | 772.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 15:15:00 | 777.00 | 777.95 | 774.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 09:15:00 | 775.85 | 777.95 | 774.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 771.20 | 776.60 | 774.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 771.20 | 776.60 | 774.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 770.90 | 775.46 | 773.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:45:00 | 776.30 | 775.74 | 774.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 770.60 | 773.46 | 773.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 770.60 | 773.46 | 773.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 756.25 | 769.47 | 771.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 09:15:00 | 761.20 | 756.58 | 760.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 761.20 | 756.58 | 760.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 761.20 | 756.58 | 760.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 761.20 | 756.58 | 760.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 761.50 | 757.56 | 760.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:30:00 | 765.80 | 757.56 | 760.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 759.55 | 757.96 | 760.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:00:00 | 759.55 | 757.96 | 760.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 753.90 | 757.15 | 759.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:15:00 | 753.35 | 757.15 | 759.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 752.95 | 750.93 | 754.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 752.20 | 751.19 | 754.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 15:00:00 | 753.50 | 751.65 | 753.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 754.00 | 752.12 | 753.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 758.55 | 752.12 | 753.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 756.05 | 752.91 | 754.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-13 11:15:00 | 762.80 | 755.60 | 755.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 762.80 | 755.60 | 755.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 768.90 | 758.26 | 756.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 14:15:00 | 816.45 | 818.08 | 805.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 15:00:00 | 816.45 | 818.08 | 805.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 804.05 | 814.78 | 805.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 804.05 | 814.78 | 805.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 805.10 | 812.84 | 805.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:15:00 | 796.30 | 812.84 | 805.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 801.05 | 810.49 | 805.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 795.50 | 810.49 | 805.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 799.75 | 808.34 | 804.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:15:00 | 798.40 | 808.34 | 804.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 803.00 | 805.36 | 804.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 811.20 | 805.36 | 804.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-25 09:15:00 | 892.32 | 871.45 | 859.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 923.60 | 961.35 | 964.57 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 994.50 | 944.33 | 941.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 1030.90 | 995.96 | 986.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 14:15:00 | 1001.65 | 1005.87 | 996.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 14:15:00 | 1001.65 | 1005.87 | 996.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1001.65 | 1005.87 | 996.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:30:00 | 1001.20 | 1005.87 | 996.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 1001.00 | 1004.89 | 996.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 1027.00 | 1004.89 | 996.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 1012.35 | 1001.86 | 998.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 992.05 | 998.24 | 998.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 992.05 | 998.24 | 998.45 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 1002.20 | 998.42 | 998.31 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 995.95 | 998.07 | 998.20 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 14:15:00 | 1003.00 | 998.47 | 998.32 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 990.05 | 997.50 | 997.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 978.95 | 993.79 | 996.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 989.40 | 985.44 | 990.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 989.40 | 985.44 | 990.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 989.40 | 985.44 | 990.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 989.40 | 985.44 | 990.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 992.20 | 986.79 | 990.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:30:00 | 995.65 | 986.79 | 990.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 990.95 | 987.62 | 990.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 992.50 | 987.62 | 990.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 991.00 | 988.30 | 990.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:45:00 | 991.90 | 988.30 | 990.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 991.00 | 988.84 | 990.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 995.55 | 988.84 | 990.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 974.40 | 985.95 | 989.21 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 15:15:00 | 997.85 | 986.93 | 986.56 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 978.00 | 985.14 | 985.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 970.40 | 982.19 | 984.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 11:15:00 | 988.05 | 983.37 | 984.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 11:15:00 | 988.05 | 983.37 | 984.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 988.05 | 983.37 | 984.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 12:00:00 | 988.05 | 983.37 | 984.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 981.30 | 982.95 | 984.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:45:00 | 968.40 | 979.52 | 982.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 983.10 | 960.21 | 960.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 983.10 | 960.21 | 960.11 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 958.90 | 965.18 | 965.37 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 13:15:00 | 970.50 | 966.19 | 965.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 09:15:00 | 979.50 | 969.65 | 967.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 15:15:00 | 972.25 | 973.51 | 970.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-06 09:15:00 | 987.50 | 973.51 | 970.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 990.25 | 976.86 | 972.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 10:15:00 | 993.75 | 976.86 | 972.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 15:15:00 | 990.45 | 1001.91 | 1002.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 990.45 | 1001.91 | 1002.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 958.95 | 989.96 | 996.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 961.35 | 956.60 | 972.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 961.35 | 956.60 | 972.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 948.15 | 931.80 | 940.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 948.15 | 931.80 | 940.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 942.85 | 934.01 | 940.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:15:00 | 942.00 | 942.76 | 943.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 972.20 | 927.70 | 927.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 972.20 | 927.70 | 927.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 1045.30 | 989.42 | 974.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 13:15:00 | 1055.05 | 1055.73 | 1038.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 13:45:00 | 1058.35 | 1055.73 | 1038.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1059.75 | 1064.69 | 1055.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 1058.65 | 1064.69 | 1055.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1060.00 | 1063.89 | 1058.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:30:00 | 1056.10 | 1063.89 | 1058.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 1059.50 | 1063.01 | 1058.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 1054.25 | 1063.01 | 1058.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1045.35 | 1059.48 | 1057.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:45:00 | 1045.65 | 1059.48 | 1057.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1049.00 | 1057.38 | 1056.54 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 1044.95 | 1054.90 | 1055.49 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 1066.55 | 1055.00 | 1054.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 1094.95 | 1062.99 | 1058.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 09:15:00 | 1109.05 | 1113.54 | 1097.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 09:45:00 | 1105.75 | 1113.54 | 1097.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1105.35 | 1114.74 | 1106.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 1105.35 | 1114.74 | 1106.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1109.90 | 1113.77 | 1106.76 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 1092.50 | 1101.77 | 1102.34 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 12:15:00 | 1109.00 | 1102.49 | 1102.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 13:15:00 | 1118.75 | 1105.74 | 1103.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 1147.95 | 1150.58 | 1139.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 09:15:00 | 1159.00 | 1150.58 | 1139.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1157.35 | 1167.83 | 1156.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 1174.15 | 1167.95 | 1157.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 15:15:00 | 1139.00 | 1154.49 | 1154.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 15:15:00 | 1139.00 | 1154.49 | 1154.57 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 09:15:00 | 1156.00 | 1154.79 | 1154.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 10:15:00 | 1176.40 | 1159.11 | 1156.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 1144.60 | 1159.82 | 1158.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 13:15:00 | 1144.60 | 1159.82 | 1158.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 1144.60 | 1159.82 | 1158.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 1144.60 | 1159.82 | 1158.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 1132.70 | 1154.40 | 1155.72 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 1167.85 | 1156.78 | 1156.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 09:15:00 | 1172.90 | 1163.00 | 1160.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 09:15:00 | 1172.50 | 1178.45 | 1171.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 1172.50 | 1178.45 | 1171.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1172.50 | 1178.45 | 1171.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:45:00 | 1176.80 | 1178.45 | 1171.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 1164.95 | 1175.75 | 1170.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 1164.95 | 1175.75 | 1170.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 1164.65 | 1173.53 | 1169.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 12:45:00 | 1167.50 | 1172.03 | 1169.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 13:15:00 | 1169.75 | 1172.03 | 1169.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 14:45:00 | 1169.00 | 1169.22 | 1168.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 1169.00 | 1168.77 | 1168.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1165.65 | 1168.91 | 1168.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:45:00 | 1165.65 | 1168.91 | 1168.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1179.55 | 1171.03 | 1169.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:30:00 | 1181.20 | 1174.60 | 1171.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 14:15:00 | 1181.75 | 1174.60 | 1171.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 1181.55 | 1176.38 | 1172.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1206.80 | 1176.90 | 1173.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 1184.60 | 1183.20 | 1178.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 1184.60 | 1183.20 | 1178.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-01-02 09:15:00 | 1284.25 | 1258.16 | 1234.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 1380.25 | 1426.88 | 1428.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 15:15:00 | 1370.00 | 1402.08 | 1415.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1405.65 | 1402.79 | 1414.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1405.65 | 1402.79 | 1414.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1405.65 | 1402.79 | 1414.85 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 1443.80 | 1421.26 | 1418.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 1461.00 | 1435.19 | 1427.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 11:15:00 | 1424.95 | 1433.82 | 1428.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 11:15:00 | 1424.95 | 1433.82 | 1428.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 1424.95 | 1433.82 | 1428.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:00:00 | 1424.95 | 1433.82 | 1428.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 1432.05 | 1433.46 | 1428.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 14:45:00 | 1442.45 | 1433.72 | 1429.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 1441.30 | 1432.97 | 1429.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 1415.00 | 1429.38 | 1428.30 | SL hit (close<static) qty=1.00 sl=1422.45 alert=retest2 |

### Cycle 66 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 1414.50 | 1426.40 | 1427.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 1396.05 | 1416.74 | 1421.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 1408.75 | 1403.02 | 1410.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1408.75 | 1403.02 | 1410.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1408.75 | 1403.02 | 1410.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 1353.35 | 1408.94 | 1410.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 1380.35 | 1388.14 | 1388.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 1376.25 | 1384.26 | 1386.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 12:15:00 | 1394.95 | 1388.31 | 1387.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 12:15:00 | 1394.95 | 1388.31 | 1387.79 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 1378.10 | 1386.20 | 1386.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1321.10 | 1371.45 | 1379.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1256.35 | 1253.01 | 1288.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:30:00 | 1274.80 | 1253.01 | 1288.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1264.75 | 1254.98 | 1271.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:30:00 | 1239.85 | 1255.26 | 1269.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:00:00 | 1258.40 | 1256.95 | 1268.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 09:45:00 | 1257.70 | 1252.98 | 1262.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 1177.86 | 1228.68 | 1240.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 1195.48 | 1228.68 | 1240.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 1194.82 | 1228.68 | 1240.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1209.90 | 1187.18 | 1207.90 | SL hit (close>ema200) qty=0.50 sl=1187.18 alert=retest2 |

### Cycle 69 — BUY (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 11:15:00 | 1205.25 | 1197.92 | 1197.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 12:15:00 | 1231.45 | 1204.62 | 1200.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 1190.85 | 1209.92 | 1205.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 1190.85 | 1209.92 | 1205.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1190.85 | 1209.92 | 1205.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 1190.85 | 1209.92 | 1205.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1196.75 | 1207.29 | 1204.27 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 1187.00 | 1199.24 | 1200.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 14:15:00 | 1178.25 | 1192.48 | 1197.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1154.05 | 1143.53 | 1161.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1154.05 | 1143.53 | 1161.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1154.05 | 1143.53 | 1161.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 1149.90 | 1143.53 | 1161.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1187.00 | 1152.22 | 1163.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1185.25 | 1152.22 | 1163.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1182.00 | 1158.18 | 1165.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 1183.40 | 1158.18 | 1165.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 1195.55 | 1174.01 | 1171.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 10:15:00 | 1209.30 | 1181.06 | 1174.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 15:15:00 | 1185.00 | 1192.09 | 1183.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 09:15:00 | 1189.35 | 1192.09 | 1183.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1184.60 | 1190.59 | 1184.04 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 11:15:00 | 1147.00 | 1178.03 | 1179.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 1137.00 | 1169.83 | 1175.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 1131.50 | 1125.99 | 1143.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 1131.50 | 1125.99 | 1143.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1098.20 | 1121.88 | 1138.29 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 1158.90 | 1135.43 | 1135.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 1176.20 | 1143.58 | 1138.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1178.60 | 1182.96 | 1170.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1178.60 | 1182.96 | 1170.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1175.00 | 1181.37 | 1170.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 1175.00 | 1181.37 | 1170.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1160.20 | 1181.36 | 1175.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:15:00 | 1152.20 | 1181.36 | 1175.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1150.00 | 1175.09 | 1173.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 1148.00 | 1175.09 | 1173.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 1163.00 | 1171.96 | 1172.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 1147.70 | 1164.80 | 1168.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 997.50 | 994.73 | 1024.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:45:00 | 998.60 | 994.73 | 1024.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 1008.25 | 990.42 | 1003.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 1008.25 | 990.42 | 1003.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 1001.15 | 992.56 | 1003.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 13:30:00 | 999.35 | 996.26 | 1003.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 1026.50 | 1002.31 | 1005.87 | SL hit (close>static) qty=1.00 sl=1008.55 alert=retest2 |

### Cycle 75 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 1033.95 | 1008.64 | 1008.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1053.85 | 1017.68 | 1012.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 15:15:00 | 1150.15 | 1154.24 | 1126.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-11 09:15:00 | 1140.00 | 1154.24 | 1126.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1133.50 | 1150.09 | 1127.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 1119.35 | 1150.09 | 1127.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1132.05 | 1146.48 | 1127.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:15:00 | 1127.45 | 1146.48 | 1127.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1129.35 | 1143.05 | 1128.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:45:00 | 1128.75 | 1143.05 | 1128.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1124.55 | 1139.35 | 1127.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 1124.55 | 1139.35 | 1127.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1127.90 | 1137.06 | 1127.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:30:00 | 1117.00 | 1137.06 | 1127.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 1142.05 | 1138.06 | 1129.03 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 14:15:00 | 1113.80 | 1127.51 | 1127.81 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 1132.85 | 1128.67 | 1128.16 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 13:15:00 | 1124.10 | 1128.02 | 1128.04 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1152.45 | 1130.63 | 1128.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 1185.50 | 1141.60 | 1133.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 1258.25 | 1263.99 | 1231.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 1258.25 | 1263.99 | 1231.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1301.60 | 1302.09 | 1288.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 1305.60 | 1302.09 | 1288.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1293.60 | 1300.04 | 1289.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 1291.75 | 1300.04 | 1289.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1288.00 | 1297.63 | 1289.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:00:00 | 1288.00 | 1297.63 | 1289.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1278.90 | 1293.89 | 1288.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 1278.90 | 1293.89 | 1288.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1271.95 | 1289.50 | 1286.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1293.00 | 1289.50 | 1286.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 1288.55 | 1289.34 | 1287.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 11:15:00 | 1275.05 | 1285.37 | 1285.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 1275.05 | 1285.37 | 1285.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 1272.30 | 1282.76 | 1284.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 1283.30 | 1276.89 | 1280.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 1283.30 | 1276.89 | 1280.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1283.30 | 1276.89 | 1280.24 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 1288.35 | 1281.95 | 1281.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 1315.45 | 1288.65 | 1284.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 1292.15 | 1295.39 | 1289.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 12:45:00 | 1291.00 | 1295.39 | 1289.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1285.55 | 1293.43 | 1288.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 1285.55 | 1293.43 | 1288.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 1290.75 | 1292.89 | 1289.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 1290.75 | 1292.89 | 1289.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 1288.00 | 1291.91 | 1289.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 1252.20 | 1291.91 | 1289.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1268.30 | 1287.19 | 1287.15 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 1265.45 | 1282.84 | 1285.18 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 1313.40 | 1285.67 | 1284.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 1337.80 | 1296.09 | 1289.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 10:15:00 | 1317.05 | 1318.32 | 1306.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 10:15:00 | 1317.05 | 1318.32 | 1306.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1317.05 | 1318.32 | 1306.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:45:00 | 1315.00 | 1318.32 | 1306.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 1309.70 | 1316.80 | 1309.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:00:00 | 1309.70 | 1316.80 | 1309.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 1324.50 | 1318.34 | 1310.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:30:00 | 1311.95 | 1318.34 | 1310.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1267.25 | 1308.39 | 1307.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1267.25 | 1308.39 | 1307.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1262.95 | 1299.30 | 1303.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 1248.50 | 1289.14 | 1298.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1165.70 | 1138.65 | 1186.65 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1146.80 | 1140.92 | 1183.32 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 1181.25 | 1148.99 | 1183.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:45:00 | 1179.95 | 1148.99 | 1183.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1182.60 | 1155.71 | 1183.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 1184.85 | 1155.71 | 1183.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1187.75 | 1162.12 | 1183.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-08 13:15:00 | 1187.75 | 1162.12 | 1183.51 | SL hit (close>ema400) qty=1.00 sl=1183.51 alert=retest1 |

### Cycle 85 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 1226.70 | 1183.10 | 1177.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1237.00 | 1204.06 | 1191.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 13:15:00 | 1293.00 | 1294.63 | 1277.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 14:00:00 | 1293.00 | 1294.63 | 1277.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1300.00 | 1316.09 | 1303.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1300.00 | 1316.09 | 1303.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1303.10 | 1313.49 | 1303.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1298.00 | 1313.49 | 1303.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1305.30 | 1311.85 | 1303.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:00:00 | 1310.70 | 1309.98 | 1303.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 15:00:00 | 1316.70 | 1311.33 | 1305.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:00:00 | 1310.30 | 1312.19 | 1306.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:45:00 | 1310.40 | 1310.34 | 1307.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1286.20 | 1309.18 | 1308.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1286.20 | 1309.18 | 1308.04 | SL hit (close<static) qty=1.00 sl=1301.40 alert=retest2 |

### Cycle 86 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1275.20 | 1302.39 | 1305.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 1216.50 | 1276.53 | 1291.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 1219.40 | 1219.30 | 1239.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-30 09:45:00 | 1222.80 | 1219.30 | 1239.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1220.20 | 1212.25 | 1225.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1220.20 | 1212.25 | 1225.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1221.10 | 1206.41 | 1215.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1221.10 | 1206.41 | 1215.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1235.80 | 1212.29 | 1216.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 1233.10 | 1212.29 | 1216.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 1222.00 | 1219.83 | 1219.78 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 1214.30 | 1219.59 | 1220.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 1205.70 | 1216.81 | 1218.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1204.50 | 1204.27 | 1210.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 1204.50 | 1204.27 | 1210.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1204.50 | 1204.27 | 1210.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 1205.50 | 1204.27 | 1210.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1212.50 | 1205.92 | 1210.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 1214.30 | 1205.92 | 1210.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 1213.20 | 1207.37 | 1211.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 1213.20 | 1207.37 | 1211.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1223.70 | 1210.64 | 1212.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1223.70 | 1210.64 | 1212.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 1226.00 | 1213.71 | 1213.51 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 1199.20 | 1213.62 | 1214.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1162.00 | 1199.60 | 1207.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1250.00 | 1193.49 | 1197.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1250.00 | 1193.49 | 1197.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1250.00 | 1193.49 | 1197.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1250.00 | 1193.49 | 1197.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1253.00 | 1205.40 | 1202.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1264.80 | 1217.28 | 1207.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 13:15:00 | 1300.00 | 1318.18 | 1298.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 13:15:00 | 1300.00 | 1318.18 | 1298.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1300.00 | 1318.18 | 1298.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 1300.00 | 1318.18 | 1298.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1313.50 | 1317.24 | 1299.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 1300.00 | 1317.24 | 1299.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1305.20 | 1313.31 | 1302.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 1305.20 | 1313.31 | 1302.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1303.20 | 1311.29 | 1302.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:30:00 | 1303.70 | 1311.29 | 1302.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 1307.80 | 1310.59 | 1302.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:30:00 | 1298.40 | 1310.59 | 1302.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 1302.90 | 1309.05 | 1302.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:00:00 | 1302.90 | 1309.05 | 1302.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 1307.00 | 1308.64 | 1303.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 1321.90 | 1307.91 | 1303.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 1303.20 | 1322.49 | 1322.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1303.20 | 1322.49 | 1322.87 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 1329.00 | 1323.10 | 1322.59 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 1316.70 | 1321.50 | 1321.93 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 1330.00 | 1323.76 | 1322.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 1349.50 | 1329.91 | 1325.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 1337.00 | 1338.00 | 1332.73 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1356.30 | 1338.00 | 1332.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1338.50 | 1346.56 | 1341.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 1338.50 | 1346.56 | 1341.51 | SL hit (close<ema400) qty=1.00 sl=1341.51 alert=retest1 |

### Cycle 96 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 1334.70 | 1373.55 | 1377.77 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 1437.90 | 1387.64 | 1381.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1476.00 | 1430.15 | 1407.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 1493.00 | 1509.59 | 1485.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 10:15:00 | 1493.00 | 1509.59 | 1485.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1493.00 | 1509.59 | 1485.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1498.20 | 1509.59 | 1485.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 1481.20 | 1500.11 | 1485.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 1481.20 | 1500.11 | 1485.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 1466.40 | 1493.37 | 1483.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:00:00 | 1466.40 | 1493.37 | 1483.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 1483.70 | 1487.09 | 1483.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 13:00:00 | 1483.70 | 1487.09 | 1483.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 1492.90 | 1488.25 | 1484.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 13:30:00 | 1483.00 | 1488.25 | 1484.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1500.00 | 1511.95 | 1503.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 1500.00 | 1511.95 | 1503.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1508.00 | 1511.16 | 1504.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 1500.50 | 1511.16 | 1504.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1509.90 | 1510.91 | 1504.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1511.70 | 1510.91 | 1504.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1519.00 | 1512.53 | 1505.94 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1478.20 | 1502.56 | 1503.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1474.50 | 1489.54 | 1495.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1486.50 | 1484.34 | 1490.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:45:00 | 1483.50 | 1484.34 | 1490.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1493.70 | 1486.22 | 1491.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 1492.40 | 1486.22 | 1491.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1482.80 | 1485.53 | 1490.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 1489.90 | 1485.53 | 1490.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1516.20 | 1489.93 | 1491.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1516.20 | 1489.93 | 1491.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1517.00 | 1495.35 | 1493.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1526.00 | 1501.48 | 1496.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 1520.00 | 1521.68 | 1512.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:45:00 | 1517.70 | 1521.68 | 1512.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1507.00 | 1518.74 | 1512.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:45:00 | 1507.40 | 1518.74 | 1512.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1504.70 | 1515.93 | 1511.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 1494.60 | 1515.93 | 1511.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 1499.00 | 1508.58 | 1508.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1494.00 | 1504.61 | 1506.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1498.00 | 1485.69 | 1493.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1498.00 | 1485.69 | 1493.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1498.00 | 1485.69 | 1493.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1498.00 | 1485.69 | 1493.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1496.00 | 1487.76 | 1493.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1496.50 | 1487.76 | 1493.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1471.80 | 1484.56 | 1491.75 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 1504.80 | 1491.47 | 1490.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1525.60 | 1501.46 | 1495.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 1504.80 | 1519.34 | 1510.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 1504.80 | 1519.34 | 1510.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1504.80 | 1519.34 | 1510.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 1504.80 | 1519.34 | 1510.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1516.40 | 1518.76 | 1511.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:15:00 | 1522.00 | 1518.76 | 1511.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:15:00 | 1520.30 | 1517.16 | 1512.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1522.90 | 1524.69 | 1517.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 1524.50 | 1534.47 | 1528.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1549.30 | 1537.93 | 1531.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:45:00 | 1540.90 | 1537.93 | 1531.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1577.00 | 1575.37 | 1564.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 1567.00 | 1575.37 | 1564.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1532.70 | 1567.45 | 1563.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1532.70 | 1567.45 | 1563.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1548.10 | 1563.58 | 1562.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 1547.00 | 1560.26 | 1560.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 1547.00 | 1560.26 | 1560.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 1532.00 | 1552.01 | 1556.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 1537.90 | 1535.23 | 1543.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:00:00 | 1537.90 | 1535.23 | 1543.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1558.20 | 1539.38 | 1543.43 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 1553.20 | 1545.75 | 1545.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 1555.00 | 1548.76 | 1547.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 1546.70 | 1549.35 | 1547.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1546.70 | 1549.35 | 1547.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1546.70 | 1549.35 | 1547.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 1538.00 | 1549.35 | 1547.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1536.90 | 1546.86 | 1546.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1536.90 | 1546.86 | 1546.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 1530.10 | 1543.51 | 1545.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 13:15:00 | 1529.50 | 1538.62 | 1542.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 15:15:00 | 1486.00 | 1484.68 | 1499.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:30:00 | 1485.60 | 1485.36 | 1498.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1514.60 | 1491.21 | 1499.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1514.60 | 1491.21 | 1499.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1537.90 | 1500.55 | 1503.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 1537.00 | 1500.55 | 1503.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 12:15:00 | 1525.00 | 1505.44 | 1505.15 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1497.70 | 1506.06 | 1507.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 1484.20 | 1496.01 | 1501.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 10:15:00 | 1497.30 | 1496.27 | 1501.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 11:00:00 | 1497.30 | 1496.27 | 1501.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1486.90 | 1486.58 | 1493.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 12:15:00 | 1474.40 | 1484.64 | 1487.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 1474.00 | 1477.28 | 1482.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:00:00 | 1474.20 | 1475.23 | 1478.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 1508.30 | 1484.93 | 1482.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 1508.30 | 1484.93 | 1482.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 12:15:00 | 1525.70 | 1493.09 | 1486.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 1528.80 | 1530.58 | 1516.29 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:15:00 | 1536.30 | 1530.58 | 1516.29 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1504.30 | 1522.58 | 1519.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1504.30 | 1522.58 | 1519.59 | SL hit (close<ema400) qty=1.00 sl=1519.59 alert=retest1 |

### Cycle 108 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 1511.10 | 1522.49 | 1523.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 1505.20 | 1519.03 | 1521.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1484.90 | 1481.15 | 1493.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 1484.90 | 1481.15 | 1493.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1483.60 | 1481.64 | 1492.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 1483.60 | 1481.64 | 1492.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1486.50 | 1482.92 | 1490.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1515.00 | 1482.92 | 1490.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1509.20 | 1488.18 | 1492.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 1518.70 | 1488.18 | 1492.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1506.00 | 1491.74 | 1493.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 1509.00 | 1491.74 | 1493.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1513.90 | 1496.17 | 1495.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1526.90 | 1512.35 | 1505.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 1502.10 | 1511.73 | 1506.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 1502.10 | 1511.73 | 1506.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1502.10 | 1511.73 | 1506.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:45:00 | 1514.90 | 1511.73 | 1506.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1501.00 | 1509.59 | 1506.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1482.50 | 1509.59 | 1506.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1478.60 | 1503.39 | 1503.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 1463.30 | 1484.57 | 1493.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 1479.10 | 1475.37 | 1486.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 1479.10 | 1475.37 | 1486.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1466.00 | 1472.19 | 1479.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 1457.00 | 1469.45 | 1477.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:00:00 | 1455.80 | 1465.25 | 1474.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1384.15 | 1406.58 | 1424.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1383.01 | 1406.58 | 1424.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 1397.70 | 1395.69 | 1414.11 | SL hit (close>ema200) qty=0.50 sl=1395.69 alert=retest2 |

### Cycle 111 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 1437.80 | 1411.17 | 1409.08 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1396.20 | 1409.28 | 1409.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 10:15:00 | 1383.50 | 1398.14 | 1402.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1381.30 | 1379.55 | 1386.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:00:00 | 1381.30 | 1379.55 | 1386.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1397.90 | 1383.53 | 1387.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1397.90 | 1383.53 | 1387.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1402.00 | 1387.23 | 1388.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 1417.60 | 1387.23 | 1388.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 1420.70 | 1393.92 | 1391.69 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 1387.40 | 1397.25 | 1398.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 1373.40 | 1392.48 | 1396.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 1309.00 | 1306.57 | 1319.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:45:00 | 1311.60 | 1306.57 | 1319.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1313.50 | 1301.95 | 1311.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1313.50 | 1301.95 | 1311.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1296.50 | 1300.86 | 1310.40 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1333.30 | 1316.34 | 1314.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1354.30 | 1328.45 | 1321.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 1330.20 | 1332.18 | 1326.19 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 14:30:00 | 1335.80 | 1332.48 | 1326.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 15:15:00 | 1341.60 | 1332.48 | 1326.88 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1316.80 | 1332.59 | 1328.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 1316.80 | 1332.59 | 1328.61 | SL hit (close<ema400) qty=1.00 sl=1328.61 alert=retest1 |

### Cycle 116 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 1305.00 | 1324.58 | 1325.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1297.30 | 1315.91 | 1321.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 1298.60 | 1298.20 | 1306.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 1299.20 | 1298.20 | 1306.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1317.70 | 1302.10 | 1307.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1321.70 | 1302.10 | 1307.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1305.50 | 1302.78 | 1307.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 1296.30 | 1307.39 | 1308.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 1301.40 | 1304.42 | 1307.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 1320.00 | 1308.33 | 1308.43 | SL hit (close>static) qty=1.00 sl=1317.70 alert=retest2 |

### Cycle 117 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 1322.80 | 1311.23 | 1309.74 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 1305.70 | 1311.09 | 1311.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 1302.80 | 1308.41 | 1310.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 1308.10 | 1296.79 | 1300.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 1308.10 | 1296.79 | 1300.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1308.10 | 1296.79 | 1300.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 1307.30 | 1296.79 | 1300.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1307.00 | 1298.83 | 1301.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1304.00 | 1298.83 | 1301.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 1322.00 | 1306.85 | 1304.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 1327.30 | 1310.94 | 1306.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 1310.60 | 1311.97 | 1308.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 12:15:00 | 1310.60 | 1311.97 | 1308.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1310.60 | 1311.97 | 1308.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 1310.60 | 1311.97 | 1308.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 1310.80 | 1311.73 | 1308.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:30:00 | 1312.10 | 1311.73 | 1308.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1317.40 | 1312.87 | 1309.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:15:00 | 1310.00 | 1312.87 | 1309.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1310.00 | 1312.29 | 1309.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 1321.20 | 1314.79 | 1311.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 14:15:00 | 1332.00 | 1315.47 | 1312.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 1326.90 | 1331.88 | 1326.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 1335.40 | 1325.16 | 1324.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1318.80 | 1325.53 | 1325.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 1321.60 | 1325.53 | 1325.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1318.10 | 1324.04 | 1324.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 1318.10 | 1324.04 | 1324.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 12:15:00 | 1316.90 | 1321.66 | 1323.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 1316.80 | 1316.37 | 1319.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 1316.80 | 1316.37 | 1319.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1317.90 | 1316.68 | 1318.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 1317.90 | 1316.68 | 1318.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1317.10 | 1316.76 | 1318.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:30:00 | 1319.60 | 1316.76 | 1318.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1321.50 | 1317.71 | 1319.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 1316.30 | 1317.71 | 1319.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1316.40 | 1317.45 | 1318.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 1308.50 | 1314.24 | 1316.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 1305.00 | 1314.24 | 1316.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 1308.30 | 1310.75 | 1313.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1243.08 | 1264.07 | 1284.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1239.75 | 1264.07 | 1284.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1242.88 | 1264.07 | 1284.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 1263.20 | 1258.84 | 1271.94 | SL hit (close>ema200) qty=0.50 sl=1258.84 alert=retest2 |

### Cycle 121 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 1280.00 | 1249.88 | 1247.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 1303.20 | 1260.55 | 1252.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 1301.00 | 1305.01 | 1293.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:15:00 | 1296.00 | 1305.01 | 1293.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1293.50 | 1303.23 | 1298.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:00:00 | 1293.50 | 1303.23 | 1298.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1303.60 | 1303.30 | 1299.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 1311.90 | 1299.83 | 1298.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 1316.90 | 1326.66 | 1327.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 1316.90 | 1326.66 | 1327.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 1311.70 | 1319.22 | 1323.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 1324.60 | 1320.29 | 1323.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 1324.60 | 1320.29 | 1323.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1324.60 | 1320.29 | 1323.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 1324.60 | 1320.29 | 1323.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1315.10 | 1319.26 | 1322.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 1337.00 | 1319.26 | 1322.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1342.60 | 1323.92 | 1324.74 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1345.10 | 1328.16 | 1326.60 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 1320.10 | 1329.76 | 1330.26 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 1350.70 | 1328.24 | 1326.39 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 10:15:00 | 1318.90 | 1328.93 | 1329.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 1316.20 | 1322.08 | 1324.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1325.20 | 1320.05 | 1322.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1325.20 | 1320.05 | 1322.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1325.20 | 1320.05 | 1322.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:45:00 | 1330.00 | 1320.05 | 1322.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1328.50 | 1321.74 | 1323.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 1330.10 | 1321.74 | 1323.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1315.00 | 1320.39 | 1322.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 1312.30 | 1317.16 | 1320.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:00:00 | 1312.20 | 1313.35 | 1317.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:00:00 | 1310.50 | 1313.75 | 1317.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1311.60 | 1317.75 | 1318.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1314.10 | 1317.02 | 1317.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 1316.10 | 1317.02 | 1317.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1310.30 | 1309.09 | 1312.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-03 12:15:00 | 1320.80 | 1314.49 | 1314.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1320.80 | 1314.49 | 1314.41 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1307.10 | 1314.39 | 1314.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1306.30 | 1312.77 | 1313.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 13:15:00 | 1315.00 | 1312.45 | 1313.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 13:15:00 | 1315.00 | 1312.45 | 1313.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1315.00 | 1312.45 | 1313.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:45:00 | 1315.00 | 1312.45 | 1313.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1316.50 | 1313.26 | 1313.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 1316.50 | 1313.26 | 1313.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 1320.00 | 1314.61 | 1314.37 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1304.00 | 1312.49 | 1313.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 1297.00 | 1309.39 | 1311.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 14:15:00 | 1315.70 | 1308.52 | 1310.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 14:15:00 | 1315.70 | 1308.52 | 1310.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1315.70 | 1308.52 | 1310.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 1315.70 | 1308.52 | 1310.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1314.00 | 1309.62 | 1310.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 1296.40 | 1309.62 | 1310.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 1318.50 | 1308.80 | 1309.72 | SL hit (close>static) qty=1.00 sl=1317.90 alert=retest2 |

### Cycle 131 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 1317.90 | 1310.62 | 1310.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 15:15:00 | 1322.70 | 1317.61 | 1315.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 1313.10 | 1317.49 | 1315.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 10:15:00 | 1313.10 | 1317.49 | 1315.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1313.10 | 1317.49 | 1315.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 1315.10 | 1317.49 | 1315.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 1316.00 | 1317.19 | 1315.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:30:00 | 1311.00 | 1317.19 | 1315.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1319.00 | 1317.55 | 1315.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 1319.40 | 1317.55 | 1315.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1319.60 | 1318.64 | 1316.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 1319.60 | 1318.64 | 1316.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1307.20 | 1316.57 | 1316.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:30:00 | 1301.00 | 1316.57 | 1316.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1309.00 | 1315.06 | 1315.48 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1316.50 | 1315.58 | 1315.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 1320.00 | 1316.46 | 1315.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 1307.20 | 1316.58 | 1316.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 1307.20 | 1316.58 | 1316.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1307.20 | 1316.58 | 1316.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1307.20 | 1316.58 | 1316.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 1289.10 | 1311.09 | 1313.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 1262.60 | 1297.70 | 1306.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 09:15:00 | 1300.00 | 1290.93 | 1300.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:00:00 | 1300.00 | 1290.93 | 1300.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1285.00 | 1289.74 | 1299.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:30:00 | 1284.70 | 1289.53 | 1298.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 1274.70 | 1285.58 | 1293.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 1220.46 | 1246.20 | 1252.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 1210.96 | 1228.63 | 1241.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1233.50 | 1197.83 | 1208.18 | SL hit (close>ema200) qty=0.50 sl=1197.83 alert=retest2 |

### Cycle 135 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1236.20 | 1216.40 | 1215.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 1242.40 | 1224.96 | 1219.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 1232.60 | 1232.73 | 1225.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:45:00 | 1232.60 | 1232.73 | 1225.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1227.20 | 1230.58 | 1225.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 1228.00 | 1230.58 | 1225.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1224.10 | 1228.49 | 1225.91 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1210.00 | 1223.28 | 1223.89 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 1239.00 | 1224.80 | 1223.82 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 1217.00 | 1223.41 | 1223.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 1209.20 | 1219.16 | 1221.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 1214.00 | 1210.76 | 1215.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 12:00:00 | 1214.00 | 1210.76 | 1215.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1208.40 | 1210.29 | 1214.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:15:00 | 1207.00 | 1210.29 | 1214.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 1215.80 | 1203.22 | 1205.65 | SL hit (close>static) qty=1.00 sl=1214.90 alert=retest2 |

### Cycle 139 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 1222.20 | 1210.03 | 1208.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 1228.10 | 1215.56 | 1211.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 1215.00 | 1215.61 | 1212.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 1215.00 | 1215.61 | 1212.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1213.20 | 1215.13 | 1212.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 1210.00 | 1215.13 | 1212.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1209.10 | 1213.92 | 1211.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 1206.90 | 1213.92 | 1211.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 1218.60 | 1214.86 | 1212.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 1208.70 | 1214.86 | 1212.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1197.50 | 1211.12 | 1211.22 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1246.80 | 1216.95 | 1213.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 11:15:00 | 1261.00 | 1236.17 | 1224.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 1272.60 | 1272.78 | 1260.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 11:00:00 | 1272.60 | 1272.78 | 1260.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1288.20 | 1284.10 | 1272.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 1274.80 | 1284.10 | 1272.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1294.70 | 1303.60 | 1295.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 1293.00 | 1303.60 | 1295.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1287.10 | 1300.30 | 1294.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:30:00 | 1284.80 | 1300.30 | 1294.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1304.00 | 1301.04 | 1295.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 1308.00 | 1301.04 | 1295.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:15:00 | 1307.00 | 1302.37 | 1297.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 1305.20 | 1302.56 | 1298.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 1307.00 | 1303.45 | 1298.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1305.10 | 1308.50 | 1304.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1322.00 | 1304.59 | 1303.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 1311.10 | 1365.63 | 1369.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1311.10 | 1365.63 | 1369.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 1285.30 | 1331.57 | 1350.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1328.40 | 1326.36 | 1344.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:30:00 | 1328.90 | 1326.36 | 1344.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1349.00 | 1330.38 | 1335.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 1349.00 | 1330.38 | 1335.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1353.10 | 1334.92 | 1337.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:30:00 | 1350.00 | 1334.92 | 1337.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 1354.10 | 1341.94 | 1340.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1373.20 | 1349.74 | 1344.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 1347.00 | 1351.36 | 1347.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 1347.00 | 1351.36 | 1347.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1347.00 | 1351.36 | 1347.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 1346.10 | 1351.36 | 1347.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1351.00 | 1351.29 | 1347.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:15:00 | 1346.40 | 1351.29 | 1347.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1346.40 | 1350.31 | 1347.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 1337.60 | 1350.31 | 1347.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1338.50 | 1347.95 | 1346.53 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 1339.00 | 1344.84 | 1345.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 1335.40 | 1341.83 | 1343.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 1330.00 | 1324.70 | 1330.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 1330.00 | 1324.70 | 1330.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1330.00 | 1324.70 | 1330.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 1330.00 | 1324.70 | 1330.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1325.10 | 1324.78 | 1330.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 1329.00 | 1324.78 | 1330.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1340.00 | 1327.83 | 1330.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:00:00 | 1340.00 | 1327.83 | 1330.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1336.90 | 1329.64 | 1331.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1346.40 | 1329.64 | 1331.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1255.30 | 1255.77 | 1270.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1250.50 | 1255.50 | 1269.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 1249.60 | 1252.62 | 1266.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1187.97 | 1197.10 | 1212.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1187.12 | 1197.10 | 1212.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 1125.45 | 1168.34 | 1192.81 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 145 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 1143.00 | 1133.80 | 1133.40 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 1108.90 | 1130.69 | 1132.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1075.30 | 1100.21 | 1111.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 1073.90 | 1073.49 | 1088.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 1073.90 | 1073.49 | 1088.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1099.70 | 1078.73 | 1089.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1099.70 | 1078.73 | 1089.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1105.70 | 1084.12 | 1091.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1134.40 | 1084.12 | 1091.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1140.20 | 1102.29 | 1098.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 1160.00 | 1113.84 | 1104.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 1262.30 | 1264.44 | 1234.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 1262.30 | 1264.44 | 1234.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1225.80 | 1256.71 | 1233.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1225.80 | 1256.71 | 1233.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1251.70 | 1255.71 | 1234.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:45:00 | 1256.50 | 1255.91 | 1236.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:00:00 | 1254.90 | 1255.18 | 1239.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:45:00 | 1258.10 | 1256.54 | 1241.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:15:00 | 1255.90 | 1259.03 | 1253.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1253.50 | 1257.93 | 1253.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:45:00 | 1253.30 | 1257.93 | 1253.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1265.10 | 1259.36 | 1254.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1233.00 | 1250.37 | 1251.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 1233.00 | 1250.37 | 1251.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 10:15:00 | 1228.50 | 1246.00 | 1249.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 09:15:00 | 1242.10 | 1234.05 | 1240.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 1242.10 | 1234.05 | 1240.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1242.10 | 1234.05 | 1240.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 1242.10 | 1234.05 | 1240.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1240.20 | 1235.28 | 1240.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:00:00 | 1235.10 | 1236.28 | 1239.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1173.34 | 1196.55 | 1212.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 1196.30 | 1190.41 | 1200.42 | SL hit (close>ema200) qty=0.50 sl=1190.41 alert=retest2 |

### Cycle 149 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 1226.10 | 1203.07 | 1200.16 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1167.80 | 1195.61 | 1198.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1155.80 | 1187.65 | 1194.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 1143.40 | 1143.19 | 1159.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1147.30 | 1144.14 | 1156.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1147.30 | 1144.14 | 1156.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1157.50 | 1144.14 | 1156.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1141.60 | 1140.11 | 1147.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:30:00 | 1142.80 | 1140.11 | 1147.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 1151.00 | 1142.99 | 1147.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:00:00 | 1151.00 | 1142.99 | 1147.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1164.10 | 1147.21 | 1148.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 1164.10 | 1147.21 | 1148.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1217.50 | 1163.00 | 1155.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 1223.10 | 1175.02 | 1161.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1248.90 | 1256.90 | 1231.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:00:00 | 1248.90 | 1256.90 | 1231.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1236.50 | 1247.69 | 1233.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 1238.90 | 1247.69 | 1233.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1230.20 | 1244.13 | 1234.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1230.20 | 1244.13 | 1234.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1237.90 | 1242.88 | 1234.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1218.00 | 1242.88 | 1234.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1245.00 | 1243.31 | 1235.55 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 1207.80 | 1228.90 | 1230.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1177.30 | 1211.89 | 1221.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 1170.40 | 1168.99 | 1185.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:00:00 | 1170.40 | 1168.99 | 1185.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1181.20 | 1171.43 | 1185.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 1182.60 | 1171.43 | 1185.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1203.80 | 1177.90 | 1186.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1203.80 | 1177.90 | 1186.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1208.90 | 1184.10 | 1188.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 1209.00 | 1189.68 | 1190.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1192.70 | 1191.01 | 1191.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:30:00 | 1194.50 | 1191.01 | 1191.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 1197.70 | 1192.35 | 1191.92 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1186.00 | 1191.15 | 1191.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1155.00 | 1183.24 | 1187.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 1184.60 | 1170.73 | 1176.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1184.60 | 1170.73 | 1176.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1184.60 | 1170.73 | 1176.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 1176.50 | 1174.39 | 1177.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:30:00 | 1177.50 | 1174.71 | 1177.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 1209.10 | 1182.13 | 1180.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1209.10 | 1182.13 | 1180.00 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1153.10 | 1181.63 | 1184.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 1146.80 | 1166.11 | 1176.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1143.00 | 1139.70 | 1154.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1143.00 | 1139.70 | 1154.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1143.00 | 1139.70 | 1154.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1152.70 | 1139.70 | 1154.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1189.90 | 1151.34 | 1157.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1189.90 | 1151.34 | 1157.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1202.50 | 1161.57 | 1161.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 1238.00 | 1197.55 | 1179.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1198.50 | 1231.05 | 1214.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1198.50 | 1231.05 | 1214.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1198.50 | 1231.05 | 1214.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1198.50 | 1231.05 | 1214.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1198.00 | 1224.44 | 1212.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 1198.90 | 1224.44 | 1212.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1185.40 | 1206.07 | 1206.36 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1252.80 | 1212.30 | 1208.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 1271.00 | 1224.04 | 1214.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 1192.20 | 1232.51 | 1225.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 1192.20 | 1232.51 | 1225.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1192.20 | 1232.51 | 1225.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1192.20 | 1232.51 | 1225.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 1195.40 | 1225.09 | 1222.88 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 1193.80 | 1218.83 | 1220.23 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 1234.10 | 1217.54 | 1217.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 1258.30 | 1228.41 | 1222.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1286.80 | 1295.66 | 1273.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 1286.50 | 1295.66 | 1273.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1279.80 | 1288.91 | 1278.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1279.80 | 1288.91 | 1278.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1275.40 | 1286.20 | 1277.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1265.60 | 1286.20 | 1277.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1261.00 | 1281.16 | 1276.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:30:00 | 1292.30 | 1279.05 | 1276.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1328.90 | 1276.02 | 1275.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-06 11:15:00 | 1421.53 | 1381.65 | 1349.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 11:15:00 | 1757.20 | 1774.36 | 1775.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 10:15:00 | 1741.90 | 1759.19 | 1766.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 09:15:00 | 1742.30 | 1741.65 | 1753.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:00:00 | 1742.30 | 1741.65 | 1753.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1746.40 | 1742.60 | 1752.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 1746.60 | 1742.60 | 1752.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 1722.20 | 1735.58 | 1744.36 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-15 09:15:00 | 705.25 | 2024-05-16 09:15:00 | 690.20 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-05-27 09:45:00 | 700.00 | 2024-06-03 11:15:00 | 693.55 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2024-05-27 10:45:00 | 700.55 | 2024-06-03 11:15:00 | 693.55 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-06-13 14:45:00 | 723.00 | 2024-06-13 15:15:00 | 718.90 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-06-14 09:30:00 | 723.85 | 2024-06-19 10:15:00 | 720.60 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-06-14 10:30:00 | 722.70 | 2024-06-19 10:15:00 | 720.60 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-06-14 14:00:00 | 725.50 | 2024-06-19 13:15:00 | 718.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-06-18 09:15:00 | 741.75 | 2024-06-19 13:15:00 | 718.00 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2024-06-19 09:30:00 | 731.70 | 2024-06-19 13:15:00 | 718.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-06-19 11:30:00 | 729.10 | 2024-06-19 13:15:00 | 718.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-07-02 09:15:00 | 743.00 | 2024-07-02 11:15:00 | 730.85 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-07-03 11:15:00 | 730.60 | 2024-07-03 13:15:00 | 741.50 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-07-11 12:15:00 | 748.60 | 2024-07-12 11:15:00 | 757.25 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-11 13:00:00 | 748.60 | 2024-07-12 11:15:00 | 757.25 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-12 14:00:00 | 748.75 | 2024-07-19 13:15:00 | 711.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 14:30:00 | 748.95 | 2024-07-19 13:15:00 | 711.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 12:45:00 | 743.50 | 2024-07-19 14:15:00 | 707.65 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2024-07-15 13:45:00 | 742.75 | 2024-07-22 09:15:00 | 706.32 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2024-07-18 11:45:00 | 743.25 | 2024-07-22 09:15:00 | 705.61 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2024-07-18 13:15:00 | 744.90 | 2024-07-22 09:15:00 | 706.09 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2024-07-18 14:15:00 | 738.95 | 2024-07-22 09:15:00 | 702.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 736.00 | 2024-07-22 09:15:00 | 699.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 14:00:00 | 748.75 | 2024-07-23 10:15:00 | 704.05 | STOP_HIT | 0.50 | 5.97% |
| SELL | retest2 | 2024-07-12 14:30:00 | 748.95 | 2024-07-23 10:15:00 | 704.05 | STOP_HIT | 0.50 | 6.00% |
| SELL | retest2 | 2024-07-15 12:45:00 | 743.50 | 2024-07-23 10:15:00 | 704.05 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2024-07-15 13:45:00 | 742.75 | 2024-07-23 10:15:00 | 704.05 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2024-07-18 11:45:00 | 743.25 | 2024-07-23 10:15:00 | 704.05 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest2 | 2024-07-18 13:15:00 | 744.90 | 2024-07-23 10:15:00 | 704.05 | STOP_HIT | 0.50 | 5.48% |
| SELL | retest2 | 2024-07-18 14:15:00 | 738.95 | 2024-07-23 10:15:00 | 704.05 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2024-07-19 09:15:00 | 736.00 | 2024-07-23 10:15:00 | 704.05 | STOP_HIT | 0.50 | 4.34% |
| BUY | retest2 | 2024-07-29 15:15:00 | 738.30 | 2024-07-31 14:15:00 | 733.15 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-07-30 10:00:00 | 738.95 | 2024-07-31 14:15:00 | 733.15 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-07-30 13:15:00 | 745.00 | 2024-07-31 14:15:00 | 733.15 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-07-30 15:15:00 | 740.00 | 2024-07-31 15:15:00 | 729.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-07-31 09:15:00 | 749.65 | 2024-07-31 15:15:00 | 729.00 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-07-31 10:15:00 | 742.45 | 2024-07-31 15:15:00 | 729.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-07-31 11:00:00 | 744.75 | 2024-07-31 15:15:00 | 729.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-08-08 10:00:00 | 730.15 | 2024-08-08 10:15:00 | 739.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-08-08 10:30:00 | 730.45 | 2024-08-08 12:15:00 | 742.65 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-08-19 10:45:00 | 736.30 | 2024-08-19 15:15:00 | 752.65 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-08-19 11:30:00 | 734.70 | 2024-08-19 15:15:00 | 752.65 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-08-23 15:00:00 | 774.00 | 2024-08-29 10:15:00 | 770.85 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-08-26 09:15:00 | 772.65 | 2024-08-29 10:15:00 | 770.85 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-08-26 11:00:00 | 774.75 | 2024-08-29 10:15:00 | 770.85 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-08-27 15:00:00 | 779.85 | 2024-08-29 10:15:00 | 770.85 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-08-28 09:15:00 | 784.05 | 2024-08-29 10:15:00 | 770.85 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-08-28 09:45:00 | 784.25 | 2024-08-29 10:15:00 | 770.85 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-09-02 09:15:00 | 751.75 | 2024-09-04 09:15:00 | 772.85 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-09-03 09:30:00 | 764.50 | 2024-09-04 09:15:00 | 772.85 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-09-03 11:00:00 | 765.60 | 2024-09-04 09:15:00 | 772.85 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-09-03 13:00:00 | 766.55 | 2024-09-04 09:15:00 | 772.85 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-09-06 11:45:00 | 776.30 | 2024-09-06 14:15:00 | 770.60 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-11 13:15:00 | 753.35 | 2024-09-13 11:15:00 | 762.80 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-09-12 13:15:00 | 752.95 | 2024-09-13 11:15:00 | 762.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-09-12 14:00:00 | 752.20 | 2024-09-13 11:15:00 | 762.80 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-09-12 15:00:00 | 753.50 | 2024-09-13 11:15:00 | 762.80 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-09-20 09:15:00 | 811.20 | 2024-09-25 09:15:00 | 892.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-16 09:15:00 | 1027.00 | 2024-10-18 11:15:00 | 992.05 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-10-17 09:15:00 | 1012.35 | 2024-10-18 11:15:00 | 992.05 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-10-25 13:45:00 | 968.40 | 2024-10-30 10:15:00 | 983.10 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-11-06 10:15:00 | 993.75 | 2024-11-11 15:15:00 | 990.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-11-19 15:15:00 | 942.00 | 2024-11-25 09:15:00 | 972.20 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-12-19 11:15:00 | 1174.15 | 2024-12-19 15:15:00 | 1139.00 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-12-26 12:45:00 | 1167.50 | 2025-01-02 09:15:00 | 1284.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-26 13:15:00 | 1169.75 | 2025-01-02 09:15:00 | 1286.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-26 14:45:00 | 1169.00 | 2025-01-02 09:15:00 | 1285.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 09:15:00 | 1169.00 | 2025-01-02 09:15:00 | 1285.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 13:30:00 | 1181.20 | 2025-01-02 09:15:00 | 1299.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 14:15:00 | 1181.75 | 2025-01-02 09:15:00 | 1299.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 14:45:00 | 1181.55 | 2025-01-02 09:15:00 | 1299.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-30 09:15:00 | 1206.80 | 2025-01-02 10:15:00 | 1327.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-06 12:45:00 | 1350.75 | 2025-01-13 12:15:00 | 1380.25 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2025-01-06 14:00:00 | 1355.85 | 2025-01-13 12:15:00 | 1380.25 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2025-01-16 14:45:00 | 1442.45 | 2025-01-17 09:15:00 | 1415.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-01-17 09:15:00 | 1441.30 | 2025-01-17 09:15:00 | 1415.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-01-22 09:15:00 | 1353.35 | 2025-01-24 12:15:00 | 1394.95 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-01-23 15:15:00 | 1380.35 | 2025-01-24 12:15:00 | 1394.95 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1376.25 | 2025-01-24 12:15:00 | 1394.95 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-01-30 10:30:00 | 1239.85 | 2025-02-03 09:15:00 | 1177.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 13:00:00 | 1258.40 | 2025-02-03 09:15:00 | 1195.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 09:45:00 | 1257.70 | 2025-02-03 09:15:00 | 1194.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 10:30:00 | 1239.85 | 2025-02-04 09:15:00 | 1209.90 | STOP_HIT | 0.50 | 2.42% |
| SELL | retest2 | 2025-01-30 13:00:00 | 1258.40 | 2025-02-04 09:15:00 | 1209.90 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2025-01-31 09:45:00 | 1257.70 | 2025-02-04 09:15:00 | 1209.90 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2025-03-05 13:30:00 | 999.35 | 2025-03-05 14:15:00 | 1026.50 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1293.00 | 2025-03-26 11:15:00 | 1275.05 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-03-26 09:45:00 | 1288.55 | 2025-03-26 11:15:00 | 1275.05 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest1 | 2025-04-08 10:30:00 | 1146.80 | 2025-04-08 13:15:00 | 1187.75 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1148.00 | 2025-04-11 10:15:00 | 1216.50 | STOP_HIT | 1.00 | -5.97% |
| BUY | retest2 | 2025-04-23 14:00:00 | 1310.70 | 2025-04-25 09:15:00 | 1286.20 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-04-23 15:00:00 | 1316.70 | 2025-04-25 09:15:00 | 1286.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-04-24 10:00:00 | 1310.30 | 2025-04-25 09:15:00 | 1286.20 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-04-24 12:45:00 | 1310.40 | 2025-04-25 09:15:00 | 1286.20 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-05-19 09:15:00 | 1321.90 | 2025-05-21 11:15:00 | 1303.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest1 | 2025-05-26 09:15:00 | 1356.30 | 2025-05-27 09:15:00 | 1338.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-05-27 13:30:00 | 1370.50 | 2025-06-02 11:15:00 | 1334.70 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-06-25 11:15:00 | 1522.00 | 2025-07-02 11:15:00 | 1547.00 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-06-25 14:15:00 | 1520.30 | 2025-07-02 11:15:00 | 1547.00 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-06-26 09:30:00 | 1522.90 | 2025-07-02 11:15:00 | 1547.00 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2025-06-27 11:30:00 | 1524.50 | 2025-07-02 11:15:00 | 1547.00 | STOP_HIT | 1.00 | 1.48% |
| SELL | retest2 | 2025-07-17 12:15:00 | 1474.40 | 2025-07-21 11:15:00 | 1508.30 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1474.00 | 2025-07-21 11:15:00 | 1508.30 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-07-18 15:00:00 | 1474.20 | 2025-07-21 11:15:00 | 1508.30 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest1 | 2025-07-23 09:15:00 | 1536.30 | 2025-07-24 09:15:00 | 1504.30 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-07-24 12:30:00 | 1537.60 | 2025-07-25 13:15:00 | 1511.10 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-07-24 15:15:00 | 1536.00 | 2025-07-25 13:15:00 | 1511.10 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-08-05 10:45:00 | 1457.00 | 2025-08-11 09:15:00 | 1384.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 13:00:00 | 1455.80 | 2025-08-11 09:15:00 | 1383.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 10:45:00 | 1457.00 | 2025-08-11 12:15:00 | 1397.70 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-08-05 13:00:00 | 1455.80 | 2025-08-11 12:15:00 | 1397.70 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-08-13 09:45:00 | 1452.40 | 2025-08-13 10:15:00 | 1437.80 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest1 | 2025-09-03 14:30:00 | 1335.80 | 2025-09-04 10:15:00 | 1316.80 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest1 | 2025-09-03 15:15:00 | 1341.60 | 2025-09-04 10:15:00 | 1316.80 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-09-08 15:15:00 | 1296.30 | 2025-09-09 11:15:00 | 1320.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-09 10:00:00 | 1301.40 | 2025-09-09 11:15:00 | 1320.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-17 13:15:00 | 1321.20 | 2025-09-22 10:15:00 | 1318.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-09-17 14:15:00 | 1332.00 | 2025-09-22 10:15:00 | 1318.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-09-19 10:00:00 | 1326.90 | 2025-09-22 10:15:00 | 1318.10 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-19 15:15:00 | 1335.40 | 2025-09-22 10:15:00 | 1318.10 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-24 13:45:00 | 1308.50 | 2025-09-26 13:15:00 | 1243.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:15:00 | 1305.00 | 2025-09-26 13:15:00 | 1239.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:30:00 | 1308.30 | 2025-09-26 13:15:00 | 1242.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 13:45:00 | 1308.50 | 2025-09-29 12:15:00 | 1263.20 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-09-24 14:15:00 | 1305.00 | 2025-09-29 12:15:00 | 1263.20 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-09-25 10:30:00 | 1308.30 | 2025-09-29 12:15:00 | 1263.20 | STOP_HIT | 0.50 | 3.45% |
| BUY | retest2 | 2025-10-09 09:15:00 | 1311.90 | 2025-10-14 09:15:00 | 1316.90 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-10-29 15:00:00 | 1312.30 | 2025-11-03 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-30 11:00:00 | 1312.20 | 2025-11-03 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-10-30 13:00:00 | 1310.50 | 2025-11-03 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1311.60 | 2025-11-03 12:15:00 | 1320.80 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1296.40 | 2025-11-07 11:15:00 | 1318.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-11-14 11:30:00 | 1284.70 | 2025-11-21 14:15:00 | 1220.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 09:30:00 | 1274.70 | 2025-11-24 11:15:00 | 1210.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 11:30:00 | 1284.70 | 2025-11-26 09:15:00 | 1233.50 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-11-17 09:30:00 | 1274.70 | 2025-11-26 09:15:00 | 1233.50 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-12-03 13:15:00 | 1207.00 | 2025-12-05 11:15:00 | 1215.80 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-12-17 15:15:00 | 1308.00 | 2025-12-30 10:15:00 | 1311.10 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2025-12-18 10:15:00 | 1307.00 | 2025-12-30 10:15:00 | 1311.10 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-12-18 12:15:00 | 1305.20 | 2025-12-30 10:15:00 | 1311.10 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-12-18 13:00:00 | 1307.00 | 2025-12-30 10:15:00 | 1311.10 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-12-22 09:15:00 | 1322.00 | 2025-12-30 10:15:00 | 1311.10 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1250.50 | 2026-01-20 13:15:00 | 1187.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:30:00 | 1249.60 | 2026-01-20 13:15:00 | 1187.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1250.50 | 2026-01-21 10:15:00 | 1125.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 11:30:00 | 1249.60 | 2026-01-22 09:15:00 | 1164.00 | STOP_HIT | 0.50 | 6.85% |
| BUY | retest2 | 2026-02-06 11:45:00 | 1256.50 | 2026-02-11 09:15:00 | 1233.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-02-06 14:00:00 | 1254.90 | 2026-02-11 09:15:00 | 1233.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-06 14:45:00 | 1258.10 | 2026-02-11 09:15:00 | 1233.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-02-10 11:15:00 | 1255.90 | 2026-02-11 09:15:00 | 1233.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-02-12 14:00:00 | 1235.10 | 2026-02-16 09:15:00 | 1173.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 14:00:00 | 1235.10 | 2026-02-17 09:15:00 | 1196.30 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2026-03-10 12:15:00 | 1176.50 | 2026-03-11 09:15:00 | 1209.10 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-03-10 13:30:00 | 1177.50 | 2026-03-11 09:15:00 | 1209.10 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2026-03-30 12:30:00 | 1292.30 | 2026-04-06 11:15:00 | 1421.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1328.90 | 2026-04-08 09:15:00 | 1461.79 | TARGET_HIT | 1.00 | 10.00% |
