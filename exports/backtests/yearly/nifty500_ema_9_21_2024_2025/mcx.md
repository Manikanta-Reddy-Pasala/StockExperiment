# Multi Commodity Exchange of India Ltd. (MCX)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 3098.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 163 |
| ALERT1 | 113 |
| ALERT2 | 110 |
| ALERT2_SKIP | 52 |
| ALERT3 | 295 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 109 |
| PARTIAL | 8 |
| TARGET_HIT | 7 |
| STOP_HIT | 106 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 92
- **Target hits / Stop hits / Partials:** 7 / 106 / 8
- **Avg / median % per leg:** -0.16% / -1.11%
- **Sum % (uncompounded):** -19.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 11 | 19.3% | 4 | 53 | 0 | -0.22% | -12.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 57 | 11 | 19.3% | 4 | 53 | 0 | -0.22% | -12.6% |
| SELL (all) | 64 | 18 | 28.1% | 3 | 53 | 8 | -0.11% | -7.2% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.91% | -7.6% |
| SELL @ 3rd Alert (retest2) | 60 | 18 | 30.0% | 3 | 49 | 8 | 0.01% | 0.5% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.91% | -7.6% |
| retest2 (combined) | 117 | 29 | 24.8% | 7 | 102 | 8 | -0.10% | -12.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 771.53 | 766.97 | 766.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 782.32 | 772.19 | 769.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 776.00 | 780.44 | 777.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 13:15:00 | 776.00 | 780.44 | 777.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 776.00 | 780.44 | 777.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 777.95 | 780.44 | 777.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 783.84 | 781.12 | 777.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 788.00 | 781.12 | 777.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:00:00 | 785.03 | 783.00 | 779.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:00:00 | 786.48 | 783.64 | 780.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 10:15:00 | 771.50 | 792.32 | 789.42 | SL hit (close<static) qty=1.00 sl=776.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 767.86 | 783.80 | 785.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 755.00 | 771.83 | 778.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 752.60 | 752.29 | 762.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 12:00:00 | 752.60 | 752.29 | 762.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 754.57 | 752.47 | 758.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:45:00 | 746.40 | 752.02 | 755.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 748.78 | 752.02 | 755.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:00:00 | 748.61 | 751.34 | 754.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 766.80 | 754.90 | 755.69 | SL hit (close>static) qty=1.00 sl=766.77 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 761.72 | 756.27 | 756.23 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 754.20 | 757.32 | 757.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 726.95 | 748.22 | 752.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 729.63 | 729.26 | 736.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 12:30:00 | 729.97 | 729.26 | 736.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 742.06 | 731.22 | 735.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:45:00 | 731.60 | 731.48 | 735.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:15:00 | 731.00 | 732.15 | 735.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:30:00 | 732.68 | 732.25 | 734.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 695.02 | 721.69 | 729.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 694.45 | 721.69 | 729.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 696.05 | 721.69 | 729.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 10:15:00 | 658.44 | 701.80 | 719.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 708.69 | 695.54 | 695.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 720.90 | 702.78 | 699.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 15:15:00 | 780.27 | 781.27 | 771.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 09:15:00 | 776.12 | 781.27 | 771.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 779.43 | 781.28 | 776.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 779.83 | 781.28 | 776.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 778.42 | 780.70 | 776.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 776.92 | 780.70 | 776.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 776.40 | 779.67 | 776.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:45:00 | 776.19 | 779.67 | 776.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 777.42 | 779.22 | 776.97 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 769.86 | 775.54 | 775.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 12:15:00 | 764.31 | 771.33 | 773.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 770.98 | 770.85 | 772.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 15:00:00 | 770.98 | 770.85 | 772.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 771.71 | 770.57 | 772.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:15:00 | 774.62 | 770.57 | 772.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 776.57 | 771.77 | 772.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 772.77 | 771.77 | 772.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 770.99 | 771.61 | 772.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:15:00 | 767.18 | 771.61 | 772.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:30:00 | 769.01 | 763.05 | 765.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 11:15:00 | 782.48 | 768.08 | 767.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 782.48 | 768.08 | 767.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 12:15:00 | 785.59 | 771.58 | 768.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 778.39 | 779.28 | 773.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 10:00:00 | 778.39 | 779.28 | 773.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 778.15 | 778.27 | 774.35 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 761.68 | 771.75 | 772.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 09:15:00 | 749.11 | 765.50 | 769.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 11:15:00 | 763.35 | 762.72 | 767.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 11:45:00 | 760.90 | 762.72 | 767.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 773.91 | 764.14 | 766.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 773.91 | 764.14 | 766.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 772.12 | 765.74 | 767.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 785.00 | 765.74 | 767.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 799.83 | 772.56 | 770.04 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 768.26 | 776.09 | 776.82 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 782.61 | 777.66 | 777.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 10:15:00 | 792.18 | 786.48 | 783.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 791.43 | 792.45 | 788.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 09:30:00 | 790.80 | 792.45 | 788.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 788.10 | 791.58 | 788.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 788.10 | 791.58 | 788.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 784.30 | 790.13 | 788.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 784.30 | 790.13 | 788.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 790.01 | 790.10 | 788.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 15:15:00 | 791.00 | 790.19 | 788.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 10:00:00 | 792.40 | 790.76 | 789.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 14:15:00 | 778.57 | 792.66 | 791.30 | SL hit (close<static) qty=1.00 sl=780.06 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 771.39 | 788.41 | 789.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 754.74 | 781.67 | 786.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 10:15:00 | 754.45 | 753.33 | 765.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 11:00:00 | 754.45 | 753.33 | 765.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 754.06 | 753.38 | 759.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:30:00 | 748.66 | 753.03 | 758.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:15:00 | 749.76 | 753.03 | 758.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 09:15:00 | 768.07 | 756.47 | 758.14 | SL hit (close>static) qty=1.00 sl=761.40 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 777.00 | 762.79 | 760.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 13:15:00 | 781.99 | 768.86 | 764.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 779.80 | 786.39 | 779.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 779.80 | 786.39 | 779.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 779.80 | 786.39 | 779.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 779.80 | 786.39 | 779.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 789.73 | 787.06 | 780.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 782.00 | 787.06 | 780.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 777.56 | 793.43 | 786.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 777.56 | 793.43 | 786.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 764.80 | 787.70 | 784.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:45:00 | 765.20 | 787.70 | 784.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 773.00 | 782.03 | 782.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 764.82 | 777.65 | 780.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 12:15:00 | 768.39 | 760.53 | 766.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 768.39 | 760.53 | 766.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 768.39 | 760.53 | 766.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:45:00 | 779.23 | 760.53 | 766.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 783.40 | 765.11 | 767.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 791.50 | 765.11 | 767.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 782.07 | 768.50 | 768.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 782.07 | 768.50 | 768.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 780.40 | 770.88 | 770.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 827.63 | 782.23 | 775.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 829.06 | 829.95 | 819.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:15:00 | 825.98 | 829.95 | 819.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 828.42 | 839.01 | 835.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:45:00 | 827.60 | 839.01 | 835.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 842.00 | 839.61 | 836.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:00:00 | 846.26 | 840.94 | 837.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 09:45:00 | 854.10 | 865.65 | 861.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 831.93 | 858.91 | 859.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 831.93 | 858.91 | 859.08 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 857.53 | 849.14 | 848.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 863.82 | 853.58 | 850.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 857.05 | 858.05 | 853.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 857.05 | 858.05 | 853.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 847.31 | 855.90 | 853.30 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 841.00 | 851.37 | 851.59 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 882.42 | 857.58 | 854.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 11:15:00 | 888.00 | 877.58 | 869.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 878.56 | 881.87 | 875.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 10:15:00 | 874.92 | 881.87 | 875.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 880.58 | 881.61 | 875.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 875.14 | 881.61 | 875.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 877.00 | 880.69 | 876.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 877.00 | 880.69 | 876.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 873.81 | 879.31 | 875.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 873.81 | 879.31 | 875.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 870.08 | 877.47 | 875.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 870.08 | 877.47 | 875.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 873.29 | 875.27 | 874.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 873.29 | 875.27 | 874.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 870.58 | 874.33 | 874.43 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 13:15:00 | 874.90 | 874.58 | 874.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 902.51 | 880.69 | 877.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 955.31 | 956.00 | 940.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 15:00:00 | 955.31 | 956.00 | 940.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 958.96 | 957.24 | 953.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 953.97 | 957.24 | 953.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 973.79 | 976.10 | 972.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:45:00 | 973.42 | 976.10 | 972.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 971.51 | 975.18 | 972.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:00:00 | 971.51 | 975.18 | 972.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 973.52 | 974.85 | 972.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:30:00 | 969.48 | 974.85 | 972.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 967.91 | 973.46 | 971.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 986.47 | 973.46 | 971.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 12:15:00 | 1047.60 | 1059.81 | 1061.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 1047.60 | 1059.81 | 1061.23 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 13:15:00 | 1063.48 | 1056.34 | 1055.56 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 13:15:00 | 1053.13 | 1055.56 | 1055.73 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1060.58 | 1056.35 | 1056.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 1066.61 | 1060.28 | 1058.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 1060.40 | 1061.06 | 1058.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:00:00 | 1060.40 | 1061.06 | 1058.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1136.46 | 1152.12 | 1136.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 1136.46 | 1152.12 | 1136.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1141.93 | 1150.09 | 1137.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 12:15:00 | 1144.54 | 1150.09 | 1137.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 13:15:00 | 1124.31 | 1141.46 | 1135.50 | SL hit (close<static) qty=1.00 sl=1127.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 15:15:00 | 1152.00 | 1168.11 | 1168.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 1129.00 | 1160.29 | 1165.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 1142.11 | 1141.16 | 1147.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 14:45:00 | 1141.00 | 1141.16 | 1147.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1138.41 | 1140.71 | 1146.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:00:00 | 1134.50 | 1138.57 | 1144.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:00:00 | 1130.24 | 1136.90 | 1142.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 1162.00 | 1143.61 | 1143.95 | SL hit (close>static) qty=1.00 sl=1154.38 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 1165.69 | 1148.02 | 1145.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 10:15:00 | 1178.89 | 1163.75 | 1155.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 12:15:00 | 1162.80 | 1165.18 | 1157.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 13:00:00 | 1162.80 | 1165.18 | 1157.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1152.26 | 1162.60 | 1157.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 1152.26 | 1162.60 | 1157.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1156.73 | 1161.43 | 1157.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:30:00 | 1172.16 | 1165.00 | 1159.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 14:45:00 | 1163.01 | 1169.04 | 1164.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1166.20 | 1167.45 | 1164.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 1150.10 | 1163.00 | 1162.52 | SL hit (close<static) qty=1.00 sl=1150.19 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 1155.92 | 1161.56 | 1161.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 1150.80 | 1159.41 | 1160.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 15:15:00 | 1158.20 | 1157.88 | 1159.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:15:00 | 1172.00 | 1157.88 | 1159.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 29 — BUY (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 09:15:00 | 1179.49 | 1162.20 | 1161.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 12:15:00 | 1190.03 | 1173.45 | 1167.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 1295.38 | 1299.92 | 1286.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 12:15:00 | 1287.40 | 1296.87 | 1288.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 1287.40 | 1296.87 | 1288.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:00:00 | 1287.40 | 1296.87 | 1288.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 1297.80 | 1297.05 | 1289.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:30:00 | 1281.43 | 1297.05 | 1289.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 1285.03 | 1294.65 | 1289.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 1285.03 | 1294.65 | 1289.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 1284.02 | 1292.52 | 1288.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 1283.60 | 1292.52 | 1288.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 1272.80 | 1285.92 | 1286.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 1268.45 | 1278.79 | 1282.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 1290.40 | 1279.93 | 1282.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 1290.40 | 1279.93 | 1282.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1290.40 | 1279.93 | 1282.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:45:00 | 1291.90 | 1279.93 | 1282.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 1292.60 | 1282.46 | 1283.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 1304.54 | 1282.46 | 1283.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 1303.00 | 1286.57 | 1285.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 13:15:00 | 1313.39 | 1295.01 | 1289.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 1299.84 | 1313.32 | 1306.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 1299.84 | 1313.32 | 1306.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 1299.84 | 1313.32 | 1306.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 1299.84 | 1313.32 | 1306.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1306.50 | 1311.96 | 1306.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 1299.30 | 1311.96 | 1306.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 1285.99 | 1306.76 | 1304.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:45:00 | 1287.88 | 1306.76 | 1304.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 1293.59 | 1304.13 | 1303.33 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 13:15:00 | 1295.74 | 1302.45 | 1302.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 1283.06 | 1298.57 | 1300.86 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 09:15:00 | 1344.60 | 1306.85 | 1304.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 11:15:00 | 1346.40 | 1320.21 | 1311.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 09:15:00 | 1329.23 | 1332.61 | 1322.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-24 10:00:00 | 1329.23 | 1332.61 | 1322.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 1310.67 | 1327.80 | 1321.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:00:00 | 1310.67 | 1327.80 | 1321.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 1319.93 | 1326.23 | 1321.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:30:00 | 1304.98 | 1326.23 | 1321.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 1316.05 | 1324.19 | 1321.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:00:00 | 1316.05 | 1324.19 | 1321.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1331.75 | 1325.70 | 1321.99 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 1287.42 | 1314.07 | 1317.37 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 1318.38 | 1308.41 | 1308.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 12:15:00 | 1335.08 | 1316.21 | 1312.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 10:15:00 | 1327.00 | 1336.90 | 1326.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 10:15:00 | 1327.00 | 1336.90 | 1326.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 1327.00 | 1336.90 | 1326.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 1327.00 | 1336.90 | 1326.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 1307.77 | 1331.07 | 1324.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:00:00 | 1307.77 | 1331.07 | 1324.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 1285.00 | 1321.86 | 1320.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:00:00 | 1285.00 | 1321.86 | 1320.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 1287.40 | 1314.97 | 1317.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 11:15:00 | 1253.60 | 1279.58 | 1291.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 13:15:00 | 1289.24 | 1280.36 | 1290.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 13:15:00 | 1289.24 | 1280.36 | 1290.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 1289.24 | 1280.36 | 1290.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:00:00 | 1289.24 | 1280.36 | 1290.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 1291.52 | 1282.59 | 1290.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 1291.52 | 1282.59 | 1290.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 1290.80 | 1284.23 | 1290.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 1290.00 | 1284.23 | 1290.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1256.35 | 1278.66 | 1287.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 10:30:00 | 1244.81 | 1271.47 | 1283.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 13:15:00 | 1298.75 | 1268.44 | 1278.03 | SL hit (close>static) qty=1.00 sl=1297.11 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1302.35 | 1285.07 | 1284.17 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1265.00 | 1282.23 | 1284.04 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 09:15:00 | 1296.78 | 1283.89 | 1283.45 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 1280.05 | 1285.08 | 1285.19 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 1288.00 | 1285.66 | 1285.44 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 1278.95 | 1284.32 | 1284.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 15:15:00 | 1278.00 | 1282.36 | 1283.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1210.78 | 1208.39 | 1228.91 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 13:45:00 | 1189.63 | 1202.16 | 1219.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:15:00 | 1184.00 | 1198.79 | 1214.66 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 12:15:00 | 1188.99 | 1190.19 | 1206.16 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1187.56 | 1188.96 | 1202.76 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1210.19 | 1190.89 | 1200.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1210.19 | 1190.89 | 1200.02 | SL hit (close>ema400) qty=1.00 sl=1200.02 alert=retest1 |

### Cycle 43 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1227.40 | 1206.05 | 1205.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 10:15:00 | 1242.98 | 1219.87 | 1212.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 14:15:00 | 1236.34 | 1238.73 | 1230.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 14:30:00 | 1238.00 | 1238.73 | 1230.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 1235.93 | 1241.88 | 1236.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:45:00 | 1235.95 | 1241.88 | 1236.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1225.39 | 1238.58 | 1235.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 1225.39 | 1238.58 | 1235.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 1222.00 | 1235.26 | 1234.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 1227.63 | 1235.26 | 1234.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 10:15:00 | 1227.30 | 1232.19 | 1232.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 1227.30 | 1232.19 | 1232.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 1205.60 | 1223.47 | 1228.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 1216.67 | 1214.63 | 1221.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 12:30:00 | 1217.45 | 1214.63 | 1221.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 1219.02 | 1215.51 | 1221.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 1221.49 | 1215.51 | 1221.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 1223.68 | 1217.14 | 1221.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:30:00 | 1225.32 | 1217.14 | 1221.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 1223.57 | 1218.43 | 1221.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 1227.35 | 1218.43 | 1221.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1235.94 | 1221.93 | 1222.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 1240.60 | 1221.93 | 1222.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 10:15:00 | 1230.00 | 1223.54 | 1223.52 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 1218.63 | 1222.97 | 1223.35 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 1233.38 | 1224.57 | 1223.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 13:15:00 | 1235.89 | 1228.59 | 1225.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 1245.37 | 1255.33 | 1245.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 1245.37 | 1255.33 | 1245.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1245.37 | 1255.33 | 1245.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 1245.37 | 1255.33 | 1245.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1237.61 | 1251.79 | 1245.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:45:00 | 1237.15 | 1251.79 | 1245.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 1231.79 | 1247.79 | 1243.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:00:00 | 1231.79 | 1247.79 | 1243.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 1244.61 | 1246.52 | 1243.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:30:00 | 1243.87 | 1246.52 | 1243.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 1241.21 | 1245.45 | 1243.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:00:00 | 1241.21 | 1245.45 | 1243.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 1244.00 | 1245.16 | 1243.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 1237.98 | 1245.16 | 1243.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1240.00 | 1244.13 | 1243.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 1245.03 | 1244.13 | 1243.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1239.48 | 1243.20 | 1243.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:00:00 | 1239.48 | 1243.20 | 1243.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1248.43 | 1244.25 | 1243.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1241.70 | 1244.25 | 1243.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1348.01 | 1365.27 | 1346.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 1348.01 | 1365.27 | 1346.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1333.19 | 1358.86 | 1345.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:00:00 | 1333.19 | 1358.86 | 1345.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 1338.84 | 1354.85 | 1344.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 12:30:00 | 1343.00 | 1352.09 | 1344.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 1341.71 | 1349.59 | 1344.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:15:00 | 1344.65 | 1343.59 | 1342.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 13:00:00 | 1344.00 | 1342.62 | 1342.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 1339.02 | 1341.90 | 1341.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:30:00 | 1338.83 | 1341.90 | 1341.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1342.02 | 1341.92 | 1341.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:30:00 | 1339.64 | 1341.92 | 1341.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-11 15:15:00 | 1340.80 | 1341.70 | 1341.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 15:15:00 | 1340.80 | 1341.70 | 1341.75 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 1357.59 | 1344.88 | 1343.19 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 1336.89 | 1342.27 | 1342.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1305.58 | 1332.74 | 1337.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 1314.55 | 1314.33 | 1323.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:45:00 | 1317.23 | 1314.33 | 1323.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 1329.24 | 1316.18 | 1322.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 1329.24 | 1316.18 | 1322.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 1345.29 | 1322.01 | 1324.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:45:00 | 1345.04 | 1322.01 | 1324.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 1339.59 | 1329.04 | 1327.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 10:15:00 | 1350.51 | 1336.35 | 1331.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 1336.46 | 1342.99 | 1337.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 1336.46 | 1342.99 | 1337.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1336.46 | 1342.99 | 1337.82 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 11:15:00 | 1335.11 | 1335.74 | 1335.80 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 1340.00 | 1336.51 | 1336.09 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 15:15:00 | 1332.60 | 1335.73 | 1335.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 1313.62 | 1331.30 | 1333.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1288.14 | 1283.12 | 1295.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 1288.14 | 1283.12 | 1295.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1288.17 | 1284.13 | 1294.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 1292.43 | 1284.13 | 1294.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1277.00 | 1270.54 | 1276.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:15:00 | 1269.62 | 1270.73 | 1275.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 1269.45 | 1270.73 | 1274.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 1266.27 | 1273.58 | 1274.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 1268.32 | 1257.36 | 1257.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 1268.32 | 1257.36 | 1257.33 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 1249.20 | 1256.85 | 1257.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 11:15:00 | 1238.60 | 1253.20 | 1255.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 1192.34 | 1183.06 | 1202.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:00:00 | 1192.34 | 1183.06 | 1202.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1203.00 | 1187.05 | 1202.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 1203.00 | 1187.05 | 1202.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1205.24 | 1190.69 | 1203.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 1205.24 | 1190.69 | 1203.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1202.00 | 1192.95 | 1203.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1182.14 | 1192.95 | 1203.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1123.03 | 1141.32 | 1159.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1128.26 | 1125.92 | 1141.24 | SL hit (close>ema200) qty=0.50 sl=1125.92 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 1183.00 | 1150.02 | 1146.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 11:15:00 | 1195.40 | 1159.09 | 1150.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 1208.60 | 1211.48 | 1194.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 1208.60 | 1211.48 | 1194.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1200.58 | 1208.40 | 1197.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 1198.38 | 1208.40 | 1197.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1190.91 | 1204.59 | 1198.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 1190.91 | 1204.59 | 1198.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1198.80 | 1203.43 | 1198.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 1202.60 | 1203.43 | 1198.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:15:00 | 1201.99 | 1200.68 | 1198.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 1116.40 | 1184.01 | 1191.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1116.40 | 1184.01 | 1191.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1075.91 | 1162.39 | 1180.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 09:15:00 | 1139.00 | 1127.97 | 1151.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 10:00:00 | 1139.00 | 1127.97 | 1151.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1143.25 | 1135.01 | 1146.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1143.25 | 1135.01 | 1146.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 1143.02 | 1136.61 | 1146.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 1159.80 | 1136.61 | 1146.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1174.00 | 1144.09 | 1148.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 1183.57 | 1144.09 | 1148.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1179.20 | 1151.11 | 1151.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1172.92 | 1151.11 | 1151.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 1177.60 | 1156.41 | 1154.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 1184.00 | 1161.93 | 1156.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 1157.90 | 1165.08 | 1160.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 1157.90 | 1165.08 | 1160.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1157.90 | 1165.08 | 1160.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 1157.90 | 1165.08 | 1160.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1157.00 | 1163.46 | 1160.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:45:00 | 1153.60 | 1163.46 | 1160.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1159.38 | 1162.65 | 1160.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:30:00 | 1154.00 | 1162.65 | 1160.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 1158.29 | 1161.78 | 1159.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:00:00 | 1158.29 | 1161.78 | 1159.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 1148.64 | 1159.15 | 1158.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 1148.64 | 1159.15 | 1158.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 1147.89 | 1156.90 | 1157.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1115.03 | 1146.94 | 1153.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 1116.46 | 1110.42 | 1124.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 1116.46 | 1110.42 | 1124.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1133.26 | 1113.75 | 1121.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1133.26 | 1113.75 | 1121.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1140.11 | 1119.02 | 1123.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:45:00 | 1144.50 | 1119.02 | 1123.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1138.56 | 1128.52 | 1127.26 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1123.00 | 1127.55 | 1127.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 1119.60 | 1124.94 | 1126.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 1134.00 | 1126.75 | 1127.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 1134.00 | 1126.75 | 1127.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1134.00 | 1126.75 | 1127.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 1134.00 | 1126.75 | 1127.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 1142.64 | 1129.93 | 1128.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 1145.09 | 1132.96 | 1130.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1140.39 | 1142.83 | 1138.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1140.39 | 1142.83 | 1138.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1140.39 | 1142.83 | 1138.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1140.60 | 1142.83 | 1138.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1156.27 | 1145.52 | 1139.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1162.39 | 1145.52 | 1139.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 13:15:00 | 1130.39 | 1139.01 | 1139.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 1130.39 | 1139.01 | 1139.50 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 1147.70 | 1139.63 | 1139.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 1171.00 | 1145.90 | 1142.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 1205.69 | 1212.77 | 1195.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 1205.69 | 1212.77 | 1195.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1207.44 | 1208.75 | 1197.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 1201.78 | 1208.75 | 1197.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1216.04 | 1210.21 | 1199.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 1220.95 | 1211.82 | 1200.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 1178.60 | 1199.99 | 1198.88 | SL hit (close<static) qty=1.00 sl=1195.27 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 1180.18 | 1196.03 | 1197.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1162.46 | 1189.32 | 1194.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1116.29 | 1106.62 | 1132.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:45:00 | 1106.31 | 1106.62 | 1132.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1132.40 | 1113.62 | 1131.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1132.40 | 1113.62 | 1131.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1109.36 | 1112.77 | 1129.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 14:30:00 | 1106.60 | 1112.42 | 1127.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 1135.21 | 1116.59 | 1127.07 | SL hit (close>static) qty=1.00 sl=1133.59 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 1122.40 | 1098.82 | 1098.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 1129.40 | 1109.66 | 1103.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1107.48 | 1129.38 | 1122.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1107.48 | 1129.38 | 1122.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1107.48 | 1129.38 | 1122.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1107.48 | 1129.38 | 1122.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1105.39 | 1124.58 | 1120.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:30:00 | 1107.36 | 1124.58 | 1120.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 1117.94 | 1121.59 | 1120.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 1112.42 | 1121.59 | 1120.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 1120.45 | 1121.36 | 1120.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 1117.30 | 1121.36 | 1120.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 1115.00 | 1120.09 | 1119.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 1099.62 | 1120.09 | 1119.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1102.17 | 1116.50 | 1118.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 1085.38 | 1101.39 | 1107.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 934.23 | 931.75 | 958.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 10:45:00 | 933.34 | 931.75 | 958.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 947.88 | 940.17 | 951.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-06 11:15:00 | 934.27 | 940.64 | 950.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 12:00:00 | 934.89 | 937.70 | 943.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 13:00:00 | 937.50 | 937.66 | 942.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:15:00 | 887.56 | 909.63 | 922.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:15:00 | 888.15 | 909.63 | 922.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:15:00 | 890.62 | 909.63 | 922.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-11 10:15:00 | 912.50 | 910.20 | 921.37 | SL hit (close>ema200) qty=0.50 sl=910.20 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 940.80 | 925.34 | 925.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 10:15:00 | 943.98 | 930.83 | 927.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 11:15:00 | 1087.00 | 1087.07 | 1070.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 12:15:00 | 1083.81 | 1087.07 | 1070.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1065.79 | 1080.12 | 1073.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1065.79 | 1080.12 | 1073.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1060.53 | 1076.21 | 1072.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 1060.53 | 1076.21 | 1072.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1058.15 | 1072.59 | 1071.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1054.00 | 1072.59 | 1071.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 1039.00 | 1065.88 | 1068.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 1031.93 | 1055.56 | 1062.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 1051.56 | 1036.51 | 1044.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 1051.56 | 1036.51 | 1044.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1051.56 | 1036.51 | 1044.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 1051.35 | 1036.51 | 1044.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1042.30 | 1037.67 | 1044.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:15:00 | 1031.77 | 1037.67 | 1044.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 1053.00 | 1042.95 | 1044.59 | SL hit (close>static) qty=1.00 sl=1051.75 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1090.00 | 1052.36 | 1048.72 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 1046.00 | 1056.14 | 1056.40 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 1066.40 | 1058.19 | 1057.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 1077.25 | 1062.00 | 1059.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1061.12 | 1069.49 | 1065.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1061.12 | 1069.49 | 1065.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1061.12 | 1069.49 | 1065.32 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1029.23 | 1058.36 | 1061.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 1021.62 | 1046.93 | 1055.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 13:15:00 | 990.14 | 988.26 | 1013.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 13:45:00 | 990.09 | 988.26 | 1013.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1051.14 | 1004.49 | 1014.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 1049.00 | 1004.49 | 1014.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 1049.72 | 1025.42 | 1022.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1073.60 | 1044.05 | 1035.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 1131.60 | 1131.90 | 1117.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 14:00:00 | 1131.60 | 1131.90 | 1117.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1195.40 | 1228.23 | 1217.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1195.40 | 1228.23 | 1217.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1208.00 | 1224.18 | 1216.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 1190.50 | 1224.18 | 1216.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 1210.30 | 1218.58 | 1215.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:45:00 | 1215.60 | 1218.58 | 1215.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 1209.00 | 1216.67 | 1215.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 09:15:00 | 1223.90 | 1216.67 | 1215.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 14:15:00 | 1224.60 | 1231.25 | 1232.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 1224.60 | 1231.25 | 1232.03 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 1269.00 | 1237.56 | 1234.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 10:15:00 | 1279.90 | 1246.03 | 1238.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 15:15:00 | 1252.90 | 1253.67 | 1246.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:15:00 | 1242.60 | 1253.67 | 1246.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1257.00 | 1254.33 | 1247.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 1249.10 | 1254.33 | 1247.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1245.00 | 1252.59 | 1247.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:45:00 | 1249.10 | 1252.59 | 1247.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1246.90 | 1251.45 | 1247.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 14:30:00 | 1251.70 | 1250.98 | 1247.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 10:15:00 | 1231.60 | 1246.69 | 1246.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 1231.60 | 1246.69 | 1246.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 1226.20 | 1242.59 | 1244.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 1213.90 | 1211.07 | 1221.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 15:00:00 | 1213.90 | 1211.07 | 1221.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1228.00 | 1215.06 | 1221.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 1228.00 | 1215.06 | 1221.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1229.90 | 1218.02 | 1222.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:15:00 | 1230.20 | 1218.02 | 1222.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 1238.30 | 1225.76 | 1225.40 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 1217.60 | 1224.12 | 1224.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 1203.30 | 1219.96 | 1222.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1206.50 | 1160.71 | 1179.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1206.50 | 1160.71 | 1179.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1206.50 | 1160.71 | 1179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1206.50 | 1160.71 | 1179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1219.60 | 1172.49 | 1183.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:30:00 | 1223.60 | 1172.49 | 1183.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1217.60 | 1194.46 | 1191.96 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 1188.40 | 1193.02 | 1193.14 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1207.20 | 1193.93 | 1193.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 1216.90 | 1198.52 | 1195.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1289.10 | 1291.43 | 1275.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1289.10 | 1291.43 | 1275.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1289.10 | 1291.43 | 1275.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1282.40 | 1291.43 | 1275.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1272.20 | 1286.19 | 1278.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1272.20 | 1286.19 | 1278.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1267.00 | 1282.35 | 1277.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 1267.00 | 1282.35 | 1277.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1279.40 | 1279.37 | 1276.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1279.40 | 1279.37 | 1276.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1264.00 | 1276.30 | 1275.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 1264.00 | 1276.30 | 1275.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1274.40 | 1275.92 | 1275.58 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 1272.10 | 1275.15 | 1275.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 1251.40 | 1269.17 | 1272.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 1277.60 | 1270.86 | 1272.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 11:15:00 | 1277.60 | 1270.86 | 1272.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1277.60 | 1270.86 | 1272.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 1277.60 | 1270.86 | 1272.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1274.00 | 1271.49 | 1272.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:15:00 | 1279.00 | 1271.49 | 1272.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1272.80 | 1271.75 | 1272.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:30:00 | 1276.70 | 1271.75 | 1272.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1273.10 | 1272.02 | 1272.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:45:00 | 1275.80 | 1272.02 | 1272.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1273.60 | 1272.34 | 1273.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1268.20 | 1272.34 | 1273.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1272.20 | 1272.31 | 1272.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 1272.20 | 1272.31 | 1272.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1278.60 | 1273.57 | 1273.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1294.60 | 1277.77 | 1275.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 1288.80 | 1290.82 | 1284.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 11:00:00 | 1288.80 | 1290.82 | 1284.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1288.90 | 1290.43 | 1284.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 1288.10 | 1290.43 | 1284.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1294.40 | 1291.23 | 1285.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 1294.60 | 1291.23 | 1285.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 1282.30 | 1289.53 | 1286.75 | SL hit (close<static) qty=1.00 sl=1285.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 1519.80 | 1534.03 | 1534.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1509.90 | 1527.83 | 1531.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 1518.10 | 1514.03 | 1519.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 1546.60 | 1514.03 | 1519.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1539.60 | 1519.15 | 1521.35 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 1542.20 | 1523.76 | 1523.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1565.60 | 1544.73 | 1534.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 1587.90 | 1588.95 | 1575.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:15:00 | 1584.40 | 1588.95 | 1575.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1569.10 | 1584.25 | 1575.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 1569.10 | 1584.25 | 1575.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1570.70 | 1581.54 | 1575.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 1559.70 | 1581.54 | 1575.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1559.70 | 1575.50 | 1573.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 1559.70 | 1575.50 | 1573.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 1554.90 | 1571.38 | 1571.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1550.50 | 1564.47 | 1568.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1586.60 | 1566.45 | 1568.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1586.60 | 1566.45 | 1568.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1586.60 | 1566.45 | 1568.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1586.60 | 1566.45 | 1568.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 1592.50 | 1571.66 | 1570.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 12:15:00 | 1605.00 | 1578.33 | 1573.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1625.80 | 1638.15 | 1617.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1625.80 | 1638.15 | 1617.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1625.80 | 1638.15 | 1617.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1616.60 | 1638.15 | 1617.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1789.00 | 1801.28 | 1792.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 1789.00 | 1801.28 | 1792.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1786.40 | 1798.30 | 1791.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1782.20 | 1798.30 | 1791.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1811.80 | 1797.82 | 1792.67 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 1777.40 | 1793.69 | 1793.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 15:15:00 | 1773.00 | 1781.38 | 1786.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1654.90 | 1636.59 | 1658.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 1654.90 | 1636.59 | 1658.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1644.00 | 1638.07 | 1657.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 1641.00 | 1639.38 | 1656.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:00:00 | 1638.60 | 1639.88 | 1653.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 1641.00 | 1640.11 | 1652.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 1667.00 | 1651.43 | 1653.59 | SL hit (close>static) qty=1.00 sl=1662.80 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1664.40 | 1655.94 | 1655.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1666.30 | 1659.43 | 1657.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 1659.50 | 1659.82 | 1657.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 1659.50 | 1659.82 | 1657.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1659.50 | 1659.82 | 1657.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 1657.20 | 1659.82 | 1657.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1658.00 | 1666.52 | 1663.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 1658.00 | 1666.52 | 1663.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1646.00 | 1662.42 | 1661.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 1646.00 | 1662.42 | 1661.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 1649.60 | 1659.85 | 1660.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1634.00 | 1651.97 | 1656.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 1650.10 | 1647.90 | 1652.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:30:00 | 1648.10 | 1647.90 | 1652.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1637.50 | 1644.75 | 1650.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1654.70 | 1644.75 | 1650.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1651.00 | 1646.20 | 1649.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1651.00 | 1646.20 | 1649.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1659.40 | 1648.84 | 1650.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 1659.00 | 1648.84 | 1650.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1648.00 | 1648.67 | 1650.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:30:00 | 1643.50 | 1649.62 | 1650.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 1640.60 | 1647.70 | 1649.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 14:15:00 | 1643.90 | 1638.38 | 1641.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 15:15:00 | 1652.00 | 1643.30 | 1643.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 1652.00 | 1643.30 | 1643.12 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 1631.90 | 1641.02 | 1642.10 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 1649.40 | 1643.74 | 1643.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 13:15:00 | 1653.30 | 1645.65 | 1644.04 | Break + close above crossover candle high |

### Cycle 96 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1622.30 | 1641.81 | 1642.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1614.30 | 1636.31 | 1640.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1566.30 | 1556.64 | 1578.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 1566.30 | 1556.64 | 1578.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1561.20 | 1561.56 | 1574.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 1558.00 | 1561.56 | 1574.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:00:00 | 1560.60 | 1561.86 | 1572.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:30:00 | 1560.30 | 1562.30 | 1570.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:30:00 | 1558.70 | 1555.30 | 1561.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1595.70 | 1543.70 | 1547.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 1595.70 | 1543.70 | 1547.35 | SL hit (close>static) qty=1.00 sl=1581.70 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 1599.60 | 1554.88 | 1552.10 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1552.20 | 1567.23 | 1567.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 1547.60 | 1560.70 | 1564.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1558.60 | 1557.77 | 1562.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:30:00 | 1554.60 | 1557.77 | 1562.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1554.30 | 1557.07 | 1561.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:30:00 | 1543.80 | 1556.47 | 1560.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1576.00 | 1560.38 | 1561.48 | SL hit (close>static) qty=1.00 sl=1563.10 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 1578.00 | 1563.90 | 1562.98 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 1551.90 | 1560.59 | 1561.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 1545.20 | 1556.02 | 1559.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 1555.60 | 1552.30 | 1556.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 1555.60 | 1552.30 | 1556.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1555.60 | 1552.30 | 1556.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 1559.20 | 1552.30 | 1556.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1566.40 | 1555.12 | 1557.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 1566.40 | 1555.12 | 1557.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 1580.00 | 1560.10 | 1559.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 1590.60 | 1570.17 | 1564.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1632.90 | 1652.40 | 1631.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 1632.90 | 1652.40 | 1631.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1632.90 | 1652.40 | 1631.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 1632.90 | 1652.40 | 1631.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1622.60 | 1646.44 | 1630.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 1622.60 | 1646.44 | 1630.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1626.20 | 1642.39 | 1630.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:15:00 | 1630.80 | 1642.39 | 1630.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:45:00 | 1632.00 | 1636.77 | 1630.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 1643.40 | 1648.90 | 1648.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 1643.40 | 1648.90 | 1648.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 1633.10 | 1644.70 | 1646.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 1612.60 | 1602.58 | 1612.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 1612.60 | 1602.58 | 1612.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1612.60 | 1602.58 | 1612.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1612.60 | 1602.58 | 1612.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1614.00 | 1604.86 | 1612.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:45:00 | 1613.00 | 1604.86 | 1612.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1611.00 | 1606.09 | 1612.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 1603.00 | 1606.43 | 1612.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1522.85 | 1547.85 | 1572.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1538.00 | 1494.74 | 1511.66 | SL hit (close>ema200) qty=0.50 sl=1494.74 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 1556.30 | 1523.33 | 1521.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 1563.60 | 1540.23 | 1530.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1536.40 | 1539.46 | 1531.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1536.40 | 1539.46 | 1531.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1536.40 | 1539.46 | 1531.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 1533.70 | 1539.46 | 1531.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1536.50 | 1541.99 | 1535.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1536.50 | 1541.99 | 1535.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1530.50 | 1539.69 | 1534.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 1528.00 | 1539.69 | 1534.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1529.30 | 1537.61 | 1534.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1511.20 | 1537.61 | 1534.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1532.20 | 1535.36 | 1533.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 1523.10 | 1535.36 | 1533.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1541.80 | 1536.65 | 1534.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:45:00 | 1532.30 | 1536.65 | 1534.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1531.80 | 1535.68 | 1534.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1531.80 | 1535.68 | 1534.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1534.00 | 1535.34 | 1534.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 1532.60 | 1535.34 | 1534.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1541.40 | 1536.55 | 1534.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 1544.70 | 1537.20 | 1535.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1530.20 | 1535.35 | 1535.30 | SL hit (close<static) qty=1.00 sl=1533.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1521.60 | 1532.60 | 1534.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1518.40 | 1529.27 | 1532.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1529.60 | 1526.08 | 1529.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1529.60 | 1526.08 | 1529.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1529.60 | 1526.08 | 1529.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1525.10 | 1526.08 | 1529.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1528.40 | 1526.54 | 1528.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 1528.40 | 1526.54 | 1528.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1536.70 | 1528.57 | 1529.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 1536.70 | 1528.57 | 1529.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 1546.00 | 1532.06 | 1531.17 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 1524.20 | 1530.20 | 1530.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 1502.90 | 1524.74 | 1528.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 13:15:00 | 1521.00 | 1519.06 | 1523.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 13:15:00 | 1521.00 | 1519.06 | 1523.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1521.00 | 1519.06 | 1523.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 1519.30 | 1519.06 | 1523.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1524.00 | 1520.52 | 1523.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1544.30 | 1520.52 | 1523.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1549.80 | 1526.38 | 1525.98 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 1524.00 | 1526.50 | 1526.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 1511.50 | 1523.50 | 1525.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 12:15:00 | 1524.10 | 1521.63 | 1523.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 1524.10 | 1521.63 | 1523.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1524.10 | 1521.63 | 1523.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 1525.50 | 1521.63 | 1523.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1520.80 | 1521.47 | 1523.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:30:00 | 1514.60 | 1519.87 | 1522.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1545.00 | 1523.99 | 1523.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1545.00 | 1523.99 | 1523.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 1555.50 | 1535.90 | 1529.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 1545.10 | 1549.36 | 1543.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1545.10 | 1549.36 | 1543.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1545.10 | 1549.36 | 1543.72 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 1523.70 | 1539.79 | 1540.38 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1576.00 | 1545.01 | 1541.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1594.80 | 1554.97 | 1546.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1603.30 | 1603.98 | 1592.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 1603.30 | 1603.98 | 1592.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1603.30 | 1603.98 | 1592.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 1615.80 | 1606.56 | 1594.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 13:00:00 | 1610.20 | 1608.29 | 1597.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 1583.40 | 1599.45 | 1596.50 | SL hit (close<static) qty=1.00 sl=1590.10 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 1589.30 | 1598.03 | 1598.31 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 1604.90 | 1599.41 | 1598.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 1616.00 | 1604.53 | 1601.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 15:15:00 | 1604.40 | 1604.52 | 1602.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 15:15:00 | 1604.40 | 1604.52 | 1602.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1604.40 | 1604.52 | 1602.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1608.60 | 1604.52 | 1602.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:45:00 | 1617.00 | 1604.53 | 1602.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 1597.80 | 1602.36 | 1601.69 | SL hit (close<static) qty=1.00 sl=1598.70 alert=retest2 |

### Cycle 114 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 1596.00 | 1601.09 | 1601.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 1581.40 | 1597.15 | 1599.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1605.40 | 1595.02 | 1597.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1605.40 | 1595.02 | 1597.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1605.40 | 1595.02 | 1597.50 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 1600.10 | 1599.14 | 1599.01 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 14:15:00 | 1587.80 | 1596.87 | 1597.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 09:15:00 | 1568.20 | 1590.04 | 1594.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 1586.10 | 1567.53 | 1575.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 1586.10 | 1567.53 | 1575.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1586.10 | 1567.53 | 1575.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 1586.10 | 1567.53 | 1575.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1582.30 | 1570.48 | 1576.48 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1599.50 | 1580.88 | 1580.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1638.70 | 1595.76 | 1587.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1640.80 | 1645.58 | 1634.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 1640.80 | 1645.58 | 1634.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1640.80 | 1645.58 | 1634.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 1635.20 | 1645.58 | 1634.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1649.60 | 1645.88 | 1636.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:45:00 | 1652.40 | 1642.94 | 1639.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 1659.90 | 1647.73 | 1641.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-14 09:15:00 | 1817.64 | 1793.58 | 1756.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 1834.00 | 1861.67 | 1864.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 1832.20 | 1855.78 | 1861.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1849.00 | 1844.12 | 1851.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 1849.00 | 1844.12 | 1851.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1849.00 | 1844.12 | 1851.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 1845.40 | 1844.12 | 1851.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1854.10 | 1846.12 | 1852.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:15:00 | 1865.00 | 1846.12 | 1852.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1857.20 | 1848.33 | 1852.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:30:00 | 1862.40 | 1848.33 | 1852.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1850.40 | 1851.41 | 1853.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:45:00 | 1855.70 | 1851.41 | 1853.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1848.50 | 1850.83 | 1852.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1841.30 | 1850.83 | 1852.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 1859.00 | 1839.19 | 1839.53 | SL hit (close>static) qty=1.00 sl=1855.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 1859.80 | 1843.31 | 1841.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 15:15:00 | 1864.20 | 1847.49 | 1843.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 1841.60 | 1846.31 | 1843.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1841.60 | 1846.31 | 1843.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1841.60 | 1846.31 | 1843.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1845.30 | 1846.31 | 1843.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1823.40 | 1841.73 | 1841.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 1823.40 | 1841.73 | 1841.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 1823.70 | 1838.12 | 1839.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 1812.70 | 1827.95 | 1833.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 1829.80 | 1825.80 | 1831.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:45:00 | 1830.40 | 1825.80 | 1831.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1824.30 | 1825.50 | 1831.07 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 1845.30 | 1829.20 | 1829.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 1852.50 | 1833.86 | 1831.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 1894.70 | 1895.42 | 1876.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:45:00 | 1888.10 | 1895.42 | 1876.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1872.70 | 1888.02 | 1877.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:45:00 | 1875.00 | 1888.02 | 1877.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1864.70 | 1883.35 | 1876.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 1864.70 | 1883.35 | 1876.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1865.60 | 1879.80 | 1875.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1862.90 | 1879.80 | 1875.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 1848.00 | 1869.78 | 1871.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 1839.40 | 1863.70 | 1868.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 1863.10 | 1855.95 | 1861.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 1863.10 | 1855.95 | 1861.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1863.10 | 1855.95 | 1861.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 1847.10 | 1855.95 | 1861.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1866.60 | 1858.08 | 1862.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 1873.60 | 1858.08 | 1862.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1868.60 | 1860.19 | 1862.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:15:00 | 1868.80 | 1860.19 | 1862.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1873.40 | 1862.83 | 1863.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 1881.00 | 1862.83 | 1863.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 1891.80 | 1868.62 | 1866.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 1904.60 | 1880.86 | 1873.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 13:15:00 | 1890.80 | 1890.82 | 1881.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:45:00 | 1885.30 | 1890.82 | 1881.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 1910.80 | 1903.85 | 1892.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1912.00 | 1899.32 | 1893.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:45:00 | 1916.00 | 1900.40 | 1894.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 1877.20 | 1894.48 | 1892.71 | SL hit (close<static) qty=1.00 sl=1892.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 14:15:00 | 1887.20 | 1891.56 | 1891.73 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 1932.30 | 1898.82 | 1894.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 1940.80 | 1911.76 | 1901.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 13:15:00 | 1926.00 | 1926.42 | 1916.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 14:00:00 | 1926.00 | 1926.42 | 1916.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1929.60 | 1937.98 | 1931.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 1951.60 | 1940.94 | 1936.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 1950.50 | 1942.21 | 1937.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:00:00 | 1954.00 | 1960.27 | 1957.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 1950.00 | 1956.97 | 1956.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 1936.40 | 1952.86 | 1954.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 1936.40 | 1952.86 | 1954.91 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 1971.00 | 1957.98 | 1956.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 1983.20 | 1972.28 | 1965.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 14:15:00 | 1971.30 | 1973.23 | 1967.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 14:15:00 | 1971.30 | 1973.23 | 1967.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1971.30 | 1973.23 | 1967.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1971.30 | 1973.23 | 1967.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2027.60 | 2059.57 | 2039.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 2027.60 | 2059.57 | 2039.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 2022.00 | 2052.06 | 2037.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:45:00 | 2025.20 | 2052.06 | 2037.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1998.40 | 2028.96 | 2029.43 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 2047.20 | 2029.01 | 2028.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 2061.80 | 2043.16 | 2036.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 14:15:00 | 2045.60 | 2047.73 | 2042.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 2045.60 | 2047.73 | 2042.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2045.60 | 2047.73 | 2042.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 2038.80 | 2047.73 | 2042.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 2045.00 | 2047.19 | 2042.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 2055.60 | 2047.19 | 2042.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 2039.80 | 2045.71 | 2042.19 | SL hit (close<static) qty=1.00 sl=2041.40 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 2018.80 | 2040.18 | 2042.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 2005.60 | 2033.26 | 2038.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 2037.00 | 2027.05 | 2032.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 2037.00 | 2027.05 | 2032.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2037.00 | 2027.05 | 2032.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:15:00 | 2038.20 | 2027.05 | 2032.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 2046.40 | 2030.92 | 2033.85 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 2051.40 | 2036.98 | 2036.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 13:15:00 | 2062.20 | 2042.03 | 2038.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 2053.60 | 2058.17 | 2049.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 2053.60 | 2058.17 | 2049.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 2046.80 | 2055.90 | 2049.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 2048.20 | 2055.90 | 2049.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 2039.80 | 2052.68 | 2048.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 2037.80 | 2052.68 | 2048.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 2024.20 | 2042.52 | 2044.35 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 2068.80 | 2047.63 | 2046.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 2074.00 | 2057.56 | 2051.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 09:15:00 | 2054.40 | 2058.69 | 2053.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 2054.40 | 2058.69 | 2053.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 2054.40 | 2058.69 | 2053.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 2051.00 | 2058.69 | 2053.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 2039.80 | 2054.92 | 2052.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 2039.80 | 2054.92 | 2052.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 2031.20 | 2050.17 | 2050.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 2031.20 | 2050.17 | 2050.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 2018.60 | 2043.86 | 2047.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 2014.40 | 2037.97 | 2044.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 2006.20 | 2002.84 | 2021.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 12:00:00 | 2006.20 | 2002.84 | 2021.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2019.60 | 1998.94 | 2011.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 2018.60 | 1998.94 | 2011.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 2009.60 | 2001.07 | 2011.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:15:00 | 2024.00 | 2001.07 | 2011.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 2033.80 | 2007.62 | 2013.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 2033.80 | 2007.62 | 2013.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 2027.80 | 2011.65 | 2014.51 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 2032.60 | 2019.03 | 2017.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 2051.40 | 2030.61 | 2023.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2012.80 | 2030.06 | 2026.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 2012.80 | 2030.06 | 2026.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2012.80 | 2030.06 | 2026.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 2012.80 | 2030.06 | 2026.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2036.80 | 2031.41 | 2027.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 2012.00 | 2031.41 | 2027.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 2029.00 | 2030.93 | 2027.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 2029.00 | 2030.93 | 2027.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 2030.20 | 2030.78 | 2027.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 2041.80 | 2030.26 | 2028.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 2039.80 | 2030.56 | 2028.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 2020.20 | 2027.76 | 2027.75 | SL hit (close<static) qty=1.00 sl=2024.20 alert=retest2 |

### Cycle 136 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 2011.00 | 2024.41 | 2026.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 2005.20 | 2018.77 | 2023.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 2015.40 | 2013.77 | 2019.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:00:00 | 2015.40 | 2013.77 | 2019.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 2030.00 | 2017.01 | 2020.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 2029.80 | 2017.01 | 2020.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2044.60 | 2022.53 | 2022.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 2044.60 | 2022.53 | 2022.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 2030.80 | 2024.18 | 2023.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 2060.20 | 2034.45 | 2028.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 2165.60 | 2167.44 | 2147.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:30:00 | 2162.80 | 2167.44 | 2147.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2189.00 | 2205.06 | 2190.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 2196.20 | 2205.06 | 2190.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2189.00 | 2201.85 | 2190.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 2177.40 | 2201.85 | 2190.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2187.00 | 2198.88 | 2189.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 2204.00 | 2198.88 | 2189.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 2219.40 | 2187.36 | 2186.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 2207.40 | 2216.43 | 2206.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 2203.00 | 2204.53 | 2203.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2245.00 | 2212.38 | 2207.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 2188.00 | 2207.42 | 2208.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 2188.00 | 2207.42 | 2208.93 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 2257.00 | 2214.62 | 2211.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 2298.00 | 2250.49 | 2233.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 2278.00 | 2289.11 | 2266.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 2278.00 | 2289.11 | 2266.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2267.00 | 2282.43 | 2267.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:30:00 | 2263.00 | 2282.43 | 2267.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 2243.00 | 2274.55 | 2265.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 2243.00 | 2274.55 | 2265.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 2230.00 | 2265.64 | 2262.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 2230.00 | 2265.64 | 2262.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 2231.00 | 2258.71 | 2259.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 2221.00 | 2244.05 | 2251.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 2227.00 | 2215.11 | 2231.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 2227.00 | 2215.11 | 2231.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2227.00 | 2215.11 | 2231.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 2227.00 | 2215.11 | 2231.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 2212.00 | 2214.49 | 2230.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 2197.00 | 2214.49 | 2230.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 2291.00 | 2239.02 | 2235.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 2291.00 | 2239.02 | 2235.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 2305.00 | 2279.09 | 2261.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 13:15:00 | 2424.00 | 2448.95 | 2415.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:00:00 | 2424.00 | 2448.95 | 2415.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 2395.00 | 2432.06 | 2415.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 2383.00 | 2432.06 | 2415.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 2391.00 | 2423.85 | 2413.52 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 2372.00 | 2407.42 | 2407.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 2349.00 | 2395.74 | 2402.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 13:15:00 | 2310.00 | 2309.48 | 2338.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 13:45:00 | 2316.00 | 2309.48 | 2338.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2323.00 | 2314.86 | 2333.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 2312.00 | 2318.95 | 2332.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:30:00 | 2313.00 | 2316.36 | 2330.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 2390.00 | 2321.51 | 2326.97 | SL hit (close>static) qty=1.00 sl=2355.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 10:15:00 | 2371.00 | 2331.41 | 2330.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 2440.00 | 2381.76 | 2358.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 2617.00 | 2647.25 | 2576.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 2617.00 | 2647.25 | 2576.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 2617.00 | 2647.25 | 2576.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 2601.00 | 2647.25 | 2576.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 2611.00 | 2640.00 | 2579.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 2594.00 | 2640.00 | 2579.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2597.00 | 2631.40 | 2581.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 2589.00 | 2631.40 | 2581.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 2589.00 | 2618.22 | 2583.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:15:00 | 2555.00 | 2618.22 | 2583.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 2546.00 | 2603.77 | 2580.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 2545.00 | 2603.77 | 2580.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2515.00 | 2586.02 | 2574.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 2275.20 | 2586.02 | 2574.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 2229.60 | 2514.73 | 2543.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2127.30 | 2387.05 | 2476.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 2253.50 | 2249.08 | 2333.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 2253.50 | 2249.08 | 2333.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2340.00 | 2278.59 | 2332.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2485.30 | 2278.59 | 2332.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2412.30 | 2305.33 | 2340.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2411.40 | 2305.33 | 2340.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 2408.30 | 2326.89 | 2346.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:00:00 | 2402.40 | 2341.99 | 2351.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2427.80 | 2369.62 | 2363.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 2427.80 | 2369.62 | 2363.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 2484.30 | 2412.48 | 2386.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2434.00 | 2492.61 | 2450.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 2434.00 | 2492.61 | 2450.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2434.00 | 2492.61 | 2450.99 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2350.10 | 2419.11 | 2428.17 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 2437.00 | 2414.94 | 2413.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 2472.80 | 2433.63 | 2423.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 2396.00 | 2446.85 | 2438.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 2396.00 | 2446.85 | 2438.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2396.00 | 2446.85 | 2438.60 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 2367.50 | 2430.98 | 2432.13 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 15:15:00 | 2438.30 | 2414.25 | 2411.88 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 2343.60 | 2400.12 | 2405.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 2278.00 | 2341.57 | 2369.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 2341.30 | 2330.04 | 2351.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 2341.30 | 2330.04 | 2351.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 2350.00 | 2334.03 | 2351.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 2311.20 | 2334.03 | 2351.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 2337.50 | 2322.71 | 2322.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 2337.50 | 2322.71 | 2322.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 2401.10 | 2338.39 | 2329.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 2359.50 | 2370.59 | 2352.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 2359.50 | 2370.59 | 2352.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 2365.50 | 2369.57 | 2353.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 2398.00 | 2369.57 | 2353.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 2525.00 | 2558.04 | 2559.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 2525.00 | 2558.04 | 2559.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2481.80 | 2540.71 | 2551.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 2534.00 | 2526.80 | 2539.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 2534.00 | 2526.80 | 2539.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 2534.00 | 2526.80 | 2539.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 2532.60 | 2526.80 | 2539.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 2496.30 | 2520.41 | 2533.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 2479.70 | 2520.41 | 2533.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 11:00:00 | 2480.90 | 2512.51 | 2528.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 12:30:00 | 2483.10 | 2501.96 | 2521.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:30:00 | 2485.80 | 2499.57 | 2518.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 2494.10 | 2498.74 | 2513.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 2507.70 | 2498.74 | 2513.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 2513.80 | 2499.49 | 2510.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 2513.80 | 2499.49 | 2510.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 2515.00 | 2502.59 | 2511.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 2515.00 | 2502.59 | 2511.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 2538.10 | 2509.69 | 2513.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:45:00 | 2536.20 | 2509.69 | 2513.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 2549.50 | 2517.66 | 2517.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 2549.50 | 2517.66 | 2517.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 2623.60 | 2545.28 | 2530.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 12:15:00 | 2609.90 | 2649.95 | 2612.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 12:15:00 | 2609.90 | 2649.95 | 2612.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 2609.90 | 2649.95 | 2612.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 2609.90 | 2649.95 | 2612.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 2634.00 | 2646.76 | 2614.12 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 2561.00 | 2595.73 | 2599.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2546.30 | 2585.84 | 2595.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 2330.00 | 2328.05 | 2385.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 2336.20 | 2328.05 | 2385.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2392.00 | 2346.99 | 2384.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 2392.00 | 2346.99 | 2384.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 2419.80 | 2361.55 | 2387.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 2419.80 | 2361.55 | 2387.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2483.00 | 2410.92 | 2405.77 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 2398.60 | 2423.28 | 2424.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2361.50 | 2407.20 | 2416.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 2410.00 | 2396.48 | 2408.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 12:15:00 | 2410.00 | 2396.48 | 2408.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 2410.00 | 2396.48 | 2408.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:00:00 | 2410.00 | 2396.48 | 2408.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 2403.00 | 2397.79 | 2407.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:30:00 | 2401.00 | 2397.79 | 2407.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 2390.50 | 2396.33 | 2406.32 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 2457.20 | 2417.11 | 2413.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 2490.30 | 2439.73 | 2425.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2371.10 | 2439.18 | 2431.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2371.10 | 2439.18 | 2431.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2371.10 | 2439.18 | 2431.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:45:00 | 2379.20 | 2439.18 | 2431.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 2378.60 | 2427.06 | 2426.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 2373.00 | 2427.06 | 2426.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 2414.90 | 2424.63 | 2425.45 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 2442.70 | 2429.06 | 2427.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 2489.00 | 2442.35 | 2433.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 2517.00 | 2521.99 | 2493.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:00:00 | 2517.00 | 2521.99 | 2493.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2680.30 | 2662.01 | 2634.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2712.00 | 2662.01 | 2634.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 15:15:00 | 2829.00 | 2840.10 | 2841.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 2829.00 | 2840.10 | 2841.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 2785.60 | 2829.20 | 2836.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 2834.40 | 2805.99 | 2817.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 2834.40 | 2805.99 | 2817.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2834.40 | 2805.99 | 2817.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 2839.10 | 2805.99 | 2817.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 2841.00 | 2812.99 | 2819.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 2841.00 | 2812.99 | 2819.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 2814.50 | 2814.78 | 2819.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 2802.00 | 2812.95 | 2817.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 2800.00 | 2812.95 | 2817.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 13:15:00 | 2827.50 | 2800.16 | 2799.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 2827.50 | 2800.16 | 2799.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 2834.50 | 2811.66 | 2805.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 2932.00 | 2943.03 | 2905.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 2932.00 | 2943.03 | 2905.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 2933.00 | 2959.21 | 2935.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 2882.90 | 2959.21 | 2935.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 2942.50 | 2955.87 | 2936.42 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 2889.80 | 2925.98 | 2927.65 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 2961.00 | 2925.39 | 2922.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 2981.00 | 2946.50 | 2933.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 15:15:00 | 788.00 | 2024-05-21 10:15:00 | 771.50 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-05-17 10:00:00 | 785.03 | 2024-05-21 10:15:00 | 771.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-05-17 12:00:00 | 786.48 | 2024-05-21 10:15:00 | 771.50 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-05-27 09:45:00 | 746.40 | 2024-05-27 13:15:00 | 766.80 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-05-27 10:15:00 | 748.78 | 2024-05-27 13:15:00 | 766.80 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-05-27 11:00:00 | 748.61 | 2024-05-27 13:15:00 | 766.80 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-06-03 10:45:00 | 731.60 | 2024-06-04 09:15:00 | 695.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:15:00 | 731.00 | 2024-06-04 09:15:00 | 694.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 13:30:00 | 732.68 | 2024-06-04 09:15:00 | 696.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:45:00 | 731.60 | 2024-06-04 10:15:00 | 658.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 12:15:00 | 731.00 | 2024-06-04 10:15:00 | 657.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 13:30:00 | 732.68 | 2024-06-04 10:15:00 | 659.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-21 12:15:00 | 767.18 | 2024-06-25 11:15:00 | 782.48 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-06-25 09:30:00 | 769.01 | 2024-06-25 11:15:00 | 782.48 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-07-08 15:15:00 | 791.00 | 2024-07-09 14:15:00 | 778.57 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-07-09 10:00:00 | 792.40 | 2024-07-09 14:15:00 | 778.57 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-07-12 11:30:00 | 748.66 | 2024-07-15 09:15:00 | 768.07 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-07-12 12:15:00 | 749.76 | 2024-07-15 09:15:00 | 768.07 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-07-31 12:00:00 | 846.26 | 2024-08-05 10:15:00 | 831.93 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-08-05 09:45:00 | 854.10 | 2024-08-05 10:15:00 | 831.93 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-08-28 09:15:00 | 986.47 | 2024-09-09 12:15:00 | 1047.60 | STOP_HIT | 1.00 | 6.20% |
| BUY | retest2 | 2024-09-19 12:15:00 | 1144.54 | 2024-09-19 13:15:00 | 1124.31 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-09-19 15:00:00 | 1147.00 | 2024-09-25 15:15:00 | 1152.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-09-30 13:00:00 | 1134.50 | 2024-10-01 10:15:00 | 1162.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-09-30 14:00:00 | 1130.24 | 2024-10-01 10:15:00 | 1162.00 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-10-04 09:30:00 | 1172.16 | 2024-10-07 10:15:00 | 1150.10 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-10-04 14:45:00 | 1163.01 | 2024-10-07 10:15:00 | 1150.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-10-07 09:15:00 | 1166.20 | 2024-10-07 10:15:00 | 1150.10 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-10-07 11:15:00 | 1165.69 | 2024-10-07 12:15:00 | 1155.92 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-11-05 10:30:00 | 1244.81 | 2024-11-05 13:15:00 | 1298.75 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest1 | 2024-11-14 13:45:00 | 1189.63 | 2024-11-19 09:15:00 | 1210.19 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest1 | 2024-11-18 09:15:00 | 1184.00 | 2024-11-19 09:15:00 | 1210.19 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest1 | 2024-11-18 12:15:00 | 1188.99 | 2024-11-19 09:15:00 | 1210.19 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest1 | 2024-11-18 14:15:00 | 1187.56 | 2024-11-19 09:15:00 | 1210.19 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-11-26 09:15:00 | 1227.63 | 2024-11-26 10:15:00 | 1227.30 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-12-10 12:30:00 | 1343.00 | 2024-12-11 15:15:00 | 1340.80 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-12-10 14:15:00 | 1341.71 | 2024-12-11 15:15:00 | 1340.80 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-12-11 10:15:00 | 1344.65 | 2024-12-11 15:15:00 | 1340.80 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-12-11 13:00:00 | 1344.00 | 2024-12-11 15:15:00 | 1340.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-12-27 12:15:00 | 1269.62 | 2025-01-02 14:15:00 | 1268.32 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-12-27 14:45:00 | 1269.45 | 2025-01-02 14:15:00 | 1268.32 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2024-12-30 12:15:00 | 1266.27 | 2025-01-02 14:15:00 | 1268.32 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1182.14 | 2025-01-13 09:15:00 | 1123.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1182.14 | 2025-01-14 09:15:00 | 1128.26 | STOP_HIT | 0.50 | 4.56% |
| BUY | retest2 | 2025-01-20 11:15:00 | 1202.60 | 2025-01-21 09:15:00 | 1116.40 | STOP_HIT | 1.00 | -7.17% |
| BUY | retest2 | 2025-01-20 14:15:00 | 1201.99 | 2025-01-21 09:15:00 | 1116.40 | STOP_HIT | 1.00 | -7.12% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1162.39 | 2025-02-03 13:15:00 | 1130.39 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-02-07 11:30:00 | 1220.95 | 2025-02-10 09:15:00 | 1178.60 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2025-02-12 14:30:00 | 1106.60 | 2025-02-13 09:15:00 | 1135.21 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1105.86 | 2025-02-19 10:15:00 | 1122.40 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-02-17 13:45:00 | 1096.86 | 2025-02-19 10:15:00 | 1122.40 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-02-18 09:15:00 | 1091.29 | 2025-02-19 10:15:00 | 1122.40 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-03-06 11:15:00 | 934.27 | 2025-03-11 09:15:00 | 887.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-07 12:00:00 | 934.89 | 2025-03-11 09:15:00 | 888.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-07 13:00:00 | 937.50 | 2025-03-11 09:15:00 | 890.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-06 11:15:00 | 934.27 | 2025-03-11 10:15:00 | 912.50 | STOP_HIT | 0.50 | 2.33% |
| SELL | retest2 | 2025-03-07 12:00:00 | 934.89 | 2025-03-11 10:15:00 | 912.50 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2025-03-07 13:00:00 | 937.50 | 2025-03-11 10:15:00 | 912.50 | STOP_HIT | 0.50 | 2.67% |
| SELL | retest2 | 2025-03-27 12:15:00 | 1031.77 | 2025-03-27 15:15:00 | 1053.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-04-28 09:15:00 | 1223.90 | 2025-04-30 14:15:00 | 1224.60 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-05-05 14:30:00 | 1251.70 | 2025-05-06 10:15:00 | 1231.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-05-26 13:15:00 | 1294.60 | 2025-05-27 09:15:00 | 1282.30 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1303.00 | 2025-06-06 09:15:00 | 1433.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1302.50 | 2025-06-06 09:15:00 | 1432.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-14 11:45:00 | 1641.00 | 2025-07-15 12:15:00 | 1667.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-07-14 14:00:00 | 1638.60 | 2025-07-15 12:15:00 | 1667.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-07-14 15:00:00 | 1641.00 | 2025-07-15 12:15:00 | 1667.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-07-22 09:30:00 | 1643.50 | 2025-07-23 15:15:00 | 1652.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-07-22 10:45:00 | 1640.60 | 2025-07-23 15:15:00 | 1652.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-23 14:15:00 | 1643.90 | 2025-07-23 15:15:00 | 1652.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-07-30 10:15:00 | 1558.00 | 2025-08-04 09:15:00 | 1595.70 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-07-30 12:00:00 | 1560.60 | 2025-08-04 09:15:00 | 1595.70 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-07-30 13:30:00 | 1560.30 | 2025-08-04 09:15:00 | 1595.70 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-07-31 13:30:00 | 1558.70 | 2025-08-04 09:15:00 | 1595.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-07 13:30:00 | 1543.80 | 2025-08-07 14:15:00 | 1576.00 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-14 12:15:00 | 1630.80 | 2025-08-20 13:15:00 | 1643.40 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-08-14 14:45:00 | 1632.00 | 2025-08-20 13:15:00 | 1643.40 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1603.00 | 2025-08-28 09:15:00 | 1522.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1603.00 | 2025-09-01 09:15:00 | 1538.00 | STOP_HIT | 0.50 | 4.05% |
| BUY | retest2 | 2025-09-04 12:45:00 | 1544.70 | 2025-09-04 14:15:00 | 1530.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-11 14:30:00 | 1514.60 | 2025-09-12 09:15:00 | 1545.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-22 10:45:00 | 1615.80 | 2025-09-23 09:15:00 | 1583.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-22 13:00:00 | 1610.20 | 2025-09-23 09:15:00 | 1583.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-09-23 14:15:00 | 1611.60 | 2025-09-24 15:15:00 | 1589.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-24 09:15:00 | 1611.10 | 2025-09-24 15:15:00 | 1589.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-26 09:15:00 | 1608.60 | 2025-09-26 11:15:00 | 1597.80 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-26 09:45:00 | 1617.00 | 2025-09-26 11:15:00 | 1597.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-09 09:45:00 | 1652.40 | 2025-10-14 09:15:00 | 1817.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-09 10:30:00 | 1659.90 | 2025-10-14 09:15:00 | 1825.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-24 09:15:00 | 1841.30 | 2025-10-27 13:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1912.00 | 2025-11-12 11:15:00 | 1877.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-12 09:45:00 | 1916.00 | 2025-11-12 11:15:00 | 1877.20 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-11-19 10:00:00 | 1951.60 | 2025-11-21 14:15:00 | 1936.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-19 11:15:00 | 1950.50 | 2025-11-21 14:15:00 | 1936.40 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-11-21 12:00:00 | 1954.00 | 2025-11-21 14:15:00 | 1936.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-21 13:45:00 | 1950.00 | 2025-11-21 14:15:00 | 1936.40 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-03 09:15:00 | 2055.60 | 2025-12-03 09:15:00 | 2039.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-03 11:00:00 | 2046.40 | 2025-12-03 14:15:00 | 2029.40 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-03 11:30:00 | 2048.80 | 2025-12-03 14:15:00 | 2029.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-04 09:30:00 | 2056.80 | 2025-12-04 10:15:00 | 2018.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-12-17 09:15:00 | 2041.80 | 2025-12-17 11:15:00 | 2020.20 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-17 09:45:00 | 2039.80 | 2025-12-17 11:15:00 | 2020.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-30 10:15:00 | 2204.00 | 2026-01-05 13:15:00 | 2188.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-12-31 09:15:00 | 2219.40 | 2026-01-05 13:15:00 | 2188.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-01-01 09:45:00 | 2207.40 | 2026-01-05 13:15:00 | 2188.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-01 15:15:00 | 2203.00 | 2026-01-05 13:15:00 | 2188.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-01-12 11:15:00 | 2197.00 | 2026-01-13 09:15:00 | 2291.00 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2026-01-23 12:00:00 | 2312.00 | 2026-01-27 09:15:00 | 2390.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2026-01-23 12:30:00 | 2313.00 | 2026-01-27 09:15:00 | 2390.00 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-02-03 10:15:00 | 2411.40 | 2026-02-03 13:15:00 | 2427.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-02-03 11:15:00 | 2408.30 | 2026-02-03 13:15:00 | 2427.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-02-03 12:00:00 | 2402.40 | 2026-02-03 13:15:00 | 2427.80 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-17 09:15:00 | 2311.20 | 2026-02-18 15:15:00 | 2337.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-20 09:15:00 | 2398.00 | 2026-03-11 14:15:00 | 2525.00 | STOP_HIT | 1.00 | 5.30% |
| SELL | retest2 | 2026-03-13 10:15:00 | 2479.70 | 2026-03-16 14:15:00 | 2549.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-03-13 11:00:00 | 2480.90 | 2026-03-16 14:15:00 | 2549.50 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-03-13 12:30:00 | 2483.10 | 2026-03-16 14:15:00 | 2549.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-03-13 13:30:00 | 2485.80 | 2026-03-16 14:15:00 | 2549.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2712.00 | 2026-04-21 15:15:00 | 2829.00 | STOP_HIT | 1.00 | 4.31% |
| SELL | retest2 | 2026-04-23 13:30:00 | 2802.00 | 2026-04-27 13:15:00 | 2827.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-04-23 14:15:00 | 2800.00 | 2026-04-27 13:15:00 | 2827.50 | STOP_HIT | 1.00 | -0.98% |
