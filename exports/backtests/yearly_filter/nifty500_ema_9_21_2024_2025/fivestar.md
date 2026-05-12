# Five-Star Business Finance Ltd. (FIVESTAR)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 462.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 138 |
| ALERT1 | 92 |
| ALERT2 | 92 |
| ALERT2_SKIP | 51 |
| ALERT3 | 249 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 101 |
| PARTIAL | 16 |
| TARGET_HIT | 9 |
| STOP_HIT | 93 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 55 / 63
- **Target hits / Stop hits / Partials:** 9 / 93 / 16
- **Avg / median % per leg:** 1.27% / -0.24%
- **Sum % (uncompounded):** 149.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 14 | 35.0% | 8 | 32 | 0 | 1.19% | 47.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.05% | -4.1% |
| BUY @ 3rd Alert (retest2) | 39 | 14 | 35.9% | 8 | 31 | 0 | 1.32% | 51.5% |
| SELL (all) | 78 | 41 | 52.6% | 1 | 61 | 16 | 1.31% | 102.1% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | -1.25% | -2.5% |
| SELL @ 3rd Alert (retest2) | 76 | 40 | 52.6% | 1 | 60 | 15 | 1.38% | 104.6% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | -2.19% | -6.6% |
| retest2 (combined) | 115 | 54 | 47.0% | 9 | 91 | 15 | 1.36% | 156.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 744.25 | 733.79 | 733.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 757.35 | 742.50 | 737.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 752.90 | 757.23 | 748.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 752.90 | 757.23 | 748.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 752.90 | 757.23 | 748.90 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 09:15:00 | 737.35 | 745.58 | 746.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 733.10 | 739.68 | 741.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 728.45 | 726.78 | 731.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 15:00:00 | 728.45 | 726.78 | 731.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 728.00 | 727.03 | 731.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 724.90 | 727.03 | 731.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 721.55 | 725.93 | 730.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:30:00 | 719.55 | 724.71 | 729.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:45:00 | 720.00 | 723.76 | 728.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 13:30:00 | 719.50 | 722.40 | 727.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 14:00:00 | 719.55 | 722.40 | 727.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 701.80 | 701.16 | 705.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-29 13:15:00 | 713.70 | 708.07 | 707.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 713.70 | 708.07 | 707.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 11:15:00 | 719.45 | 712.47 | 710.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 14:15:00 | 710.95 | 712.71 | 710.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 14:15:00 | 710.95 | 712.71 | 710.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 710.95 | 712.71 | 710.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:45:00 | 706.25 | 712.71 | 710.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 711.35 | 712.44 | 710.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 721.40 | 712.44 | 710.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-07 09:15:00 | 793.54 | 780.37 | 764.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 12:15:00 | 807.65 | 814.88 | 815.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 798.40 | 810.47 | 813.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 14:15:00 | 807.40 | 794.43 | 799.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 14:15:00 | 807.40 | 794.43 | 799.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 807.40 | 794.43 | 799.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 807.40 | 794.43 | 799.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 800.00 | 795.55 | 799.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 819.00 | 795.55 | 799.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 810.50 | 801.49 | 801.43 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 795.00 | 801.00 | 801.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 15:15:00 | 790.00 | 797.34 | 799.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 14:15:00 | 796.40 | 794.43 | 796.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 796.40 | 794.43 | 796.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 791.70 | 793.88 | 796.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 800.00 | 793.88 | 796.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 821.10 | 799.33 | 798.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 829.30 | 816.79 | 809.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 10:15:00 | 824.20 | 825.53 | 819.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 11:00:00 | 824.20 | 825.53 | 819.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 824.25 | 824.92 | 819.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:30:00 | 821.80 | 824.92 | 819.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 827.55 | 825.16 | 820.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 823.10 | 825.16 | 820.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 809.85 | 821.91 | 820.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 809.85 | 821.91 | 820.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 807.70 | 819.07 | 819.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 809.80 | 819.07 | 819.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 11:15:00 | 807.35 | 816.73 | 817.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 10:15:00 | 803.85 | 808.86 | 812.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 795.85 | 787.55 | 794.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 795.85 | 787.55 | 794.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 795.85 | 787.55 | 794.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 797.85 | 787.55 | 794.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 800.25 | 790.09 | 795.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:30:00 | 803.65 | 790.09 | 795.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 806.60 | 793.39 | 796.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:45:00 | 805.40 | 793.39 | 796.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 789.30 | 794.40 | 796.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:15:00 | 796.00 | 794.40 | 796.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 796.00 | 794.72 | 796.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 812.70 | 794.72 | 796.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 09:15:00 | 821.25 | 800.02 | 798.49 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 795.25 | 809.92 | 811.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 09:15:00 | 783.00 | 798.69 | 803.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 13:15:00 | 764.85 | 762.93 | 772.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 14:00:00 | 764.85 | 762.93 | 772.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 765.60 | 763.46 | 771.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 09:15:00 | 760.00 | 765.00 | 771.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 760.50 | 764.56 | 770.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:00:00 | 759.70 | 762.53 | 767.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 749.95 | 762.27 | 766.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:15:00 | 722.00 | 746.08 | 753.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:15:00 | 722.48 | 746.08 | 753.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:15:00 | 721.72 | 746.08 | 753.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 749.80 | 736.69 | 743.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 749.80 | 736.69 | 743.64 | SL hit (close>ema200) qty=0.50 sl=736.69 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 757.70 | 747.63 | 746.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 759.50 | 750.00 | 748.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 763.60 | 763.93 | 758.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 763.60 | 763.93 | 758.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 754.05 | 761.54 | 758.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:00:00 | 754.05 | 761.54 | 758.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 755.05 | 760.24 | 758.13 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 09:15:00 | 747.10 | 756.26 | 756.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 13:15:00 | 739.95 | 749.95 | 753.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 14:15:00 | 758.40 | 751.64 | 753.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 14:15:00 | 758.40 | 751.64 | 753.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 758.40 | 751.64 | 753.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:45:00 | 763.95 | 751.64 | 753.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 756.80 | 752.67 | 754.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 764.90 | 752.67 | 754.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 770.70 | 756.28 | 755.53 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 738.30 | 758.36 | 761.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 730.00 | 752.69 | 758.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 15:15:00 | 725.00 | 724.50 | 730.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-08 09:15:00 | 721.10 | 724.50 | 730.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 724.15 | 724.43 | 730.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 719.30 | 724.25 | 727.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 710.45 | 701.29 | 700.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 710.45 | 701.29 | 700.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 713.50 | 708.24 | 704.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 11:15:00 | 707.20 | 708.13 | 705.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 11:15:00 | 707.20 | 708.13 | 705.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 707.20 | 708.13 | 705.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:45:00 | 705.30 | 708.13 | 705.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 708.80 | 708.26 | 705.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 13:30:00 | 713.15 | 710.44 | 708.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:30:00 | 716.50 | 713.90 | 710.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 750.00 | 757.39 | 758.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 750.00 | 757.39 | 758.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 13:15:00 | 737.95 | 748.06 | 752.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 14:15:00 | 746.75 | 741.80 | 745.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 14:15:00 | 746.75 | 741.80 | 745.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 746.75 | 741.80 | 745.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 746.75 | 741.80 | 745.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 744.00 | 742.24 | 745.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 735.50 | 742.24 | 745.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:15:00 | 739.05 | 743.19 | 745.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:00:00 | 743.20 | 743.45 | 745.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 743.60 | 742.86 | 744.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 744.45 | 743.18 | 744.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 744.45 | 743.18 | 744.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 747.65 | 744.01 | 744.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 747.65 | 744.01 | 744.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 748.80 | 744.97 | 745.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:45:00 | 747.55 | 744.97 | 745.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 744.60 | 744.89 | 745.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 741.65 | 744.92 | 745.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 10:00:00 | 741.55 | 742.71 | 743.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 12:15:00 | 749.00 | 745.10 | 744.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 12:15:00 | 749.00 | 745.10 | 744.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 13:15:00 | 750.55 | 746.19 | 745.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 12:15:00 | 760.25 | 761.55 | 756.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 12:30:00 | 761.60 | 761.55 | 756.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 768.10 | 765.19 | 761.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 761.60 | 765.19 | 761.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 764.40 | 766.25 | 762.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 764.55 | 766.25 | 762.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 761.80 | 765.36 | 762.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 764.05 | 765.36 | 762.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 768.75 | 766.04 | 763.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 12:15:00 | 770.20 | 766.04 | 763.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 13:15:00 | 769.80 | 766.39 | 763.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 772.85 | 767.63 | 764.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 10:00:00 | 777.25 | 770.39 | 766.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 793.95 | 793.66 | 788.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 793.95 | 793.66 | 788.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 777.20 | 789.80 | 787.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 776.80 | 789.80 | 787.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 788.30 | 789.50 | 787.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:45:00 | 778.55 | 789.50 | 787.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 775.00 | 786.60 | 786.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:45:00 | 775.25 | 786.60 | 786.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 771.00 | 783.48 | 784.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 14:15:00 | 771.00 | 783.48 | 784.95 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 09:15:00 | 812.95 | 788.64 | 787.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 11:15:00 | 819.00 | 799.09 | 792.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 808.20 | 809.15 | 801.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 13:15:00 | 803.65 | 807.53 | 802.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 803.65 | 807.53 | 802.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:00:00 | 803.65 | 807.53 | 802.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 796.90 | 805.41 | 802.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:30:00 | 806.05 | 804.95 | 802.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 09:15:00 | 785.15 | 802.55 | 803.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 785.15 | 802.55 | 803.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 780.30 | 791.55 | 797.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 791.50 | 788.48 | 794.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 10:00:00 | 791.50 | 788.48 | 794.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 792.35 | 789.51 | 793.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:30:00 | 792.65 | 789.51 | 793.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 794.00 | 790.40 | 793.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 795.00 | 790.40 | 793.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 801.15 | 792.55 | 794.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 801.15 | 792.55 | 794.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 14:15:00 | 826.00 | 799.24 | 797.23 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 792.55 | 805.51 | 807.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 785.70 | 801.55 | 805.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 795.05 | 793.65 | 798.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 795.05 | 793.65 | 798.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 797.45 | 794.41 | 798.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 795.55 | 794.41 | 798.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 806.90 | 797.01 | 799.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 806.90 | 797.01 | 799.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 811.25 | 799.86 | 800.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 811.25 | 799.86 | 800.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 805.75 | 801.04 | 800.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 813.10 | 804.26 | 802.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 803.45 | 805.89 | 804.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 14:15:00 | 803.45 | 805.89 | 804.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 803.45 | 805.89 | 804.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 803.45 | 805.89 | 804.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 809.00 | 806.51 | 804.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:00:00 | 812.05 | 806.99 | 805.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 12:45:00 | 813.50 | 809.29 | 807.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 14:00:00 | 812.55 | 809.94 | 807.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 14:30:00 | 814.20 | 813.01 | 809.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-15 15:15:00 | 893.25 | 869.44 | 844.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 867.45 | 882.02 | 883.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 857.35 | 870.34 | 875.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 15:15:00 | 860.95 | 858.31 | 866.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 09:15:00 | 857.60 | 858.31 | 866.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 861.70 | 858.98 | 866.20 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 10:15:00 | 878.20 | 865.97 | 865.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 11:15:00 | 928.70 | 878.51 | 871.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 14:15:00 | 880.75 | 882.69 | 875.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-24 15:00:00 | 880.75 | 882.69 | 875.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 889.75 | 884.15 | 877.30 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 866.25 | 876.34 | 877.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 14:15:00 | 855.05 | 865.53 | 871.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 11:15:00 | 872.40 | 861.93 | 866.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 11:15:00 | 872.40 | 861.93 | 866.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 872.40 | 861.93 | 866.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:00:00 | 872.40 | 861.93 | 866.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 864.70 | 862.48 | 866.76 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 878.00 | 870.74 | 869.93 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 09:15:00 | 781.10 | 852.81 | 861.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 764.50 | 797.31 | 827.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 653.95 | 653.87 | 664.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:00:00 | 653.95 | 653.87 | 664.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 642.00 | 640.25 | 644.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 631.80 | 643.90 | 644.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:45:00 | 636.35 | 639.56 | 641.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:30:00 | 636.20 | 638.95 | 641.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:30:00 | 635.05 | 640.30 | 641.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 637.75 | 637.59 | 639.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 637.75 | 637.59 | 639.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 637.50 | 637.57 | 639.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:15:00 | 632.05 | 637.57 | 639.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 14:15:00 | 643.45 | 638.19 | 638.60 | SL hit (close>static) qty=1.00 sl=640.35 alert=retest2 |

### Cycle 29 — BUY (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 13:15:00 | 640.75 | 637.78 | 637.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 648.50 | 639.92 | 638.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 13:15:00 | 640.00 | 644.10 | 641.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 13:15:00 | 640.00 | 644.10 | 641.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 640.00 | 644.10 | 641.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:00:00 | 640.00 | 644.10 | 641.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 638.55 | 642.99 | 641.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 638.55 | 642.99 | 641.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 641.00 | 642.59 | 641.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 634.75 | 642.59 | 641.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 638.00 | 641.67 | 641.19 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 635.50 | 640.44 | 640.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 12:15:00 | 634.75 | 638.41 | 639.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 14:15:00 | 645.30 | 639.18 | 639.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 14:15:00 | 645.30 | 639.18 | 639.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 645.30 | 639.18 | 639.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 645.30 | 639.18 | 639.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 646.65 | 640.67 | 640.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 654.90 | 643.52 | 641.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 15:15:00 | 654.55 | 655.68 | 652.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 09:15:00 | 653.25 | 655.68 | 652.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 658.05 | 656.16 | 652.62 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 637.05 | 648.63 | 649.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 13:15:00 | 632.00 | 638.67 | 643.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 09:15:00 | 643.80 | 634.33 | 636.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 643.80 | 634.33 | 636.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 643.80 | 634.33 | 636.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:30:00 | 644.15 | 634.33 | 636.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 653.60 | 638.18 | 638.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 653.60 | 638.18 | 638.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 650.60 | 640.67 | 639.56 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 635.10 | 640.90 | 641.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 10:15:00 | 633.80 | 638.68 | 640.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 633.30 | 632.50 | 634.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 633.30 | 632.50 | 634.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 633.30 | 632.50 | 634.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:30:00 | 637.30 | 632.50 | 634.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 633.35 | 632.29 | 634.43 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 643.85 | 635.83 | 635.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 12:15:00 | 650.15 | 641.72 | 638.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 637.00 | 644.18 | 640.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 637.00 | 644.18 | 640.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 637.00 | 644.18 | 640.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 637.00 | 644.18 | 640.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 643.40 | 644.03 | 641.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 639.80 | 644.03 | 641.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 644.70 | 644.16 | 641.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:30:00 | 642.40 | 644.16 | 641.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 645.50 | 644.78 | 642.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:30:00 | 643.50 | 644.78 | 642.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 670.00 | 651.38 | 646.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 12:45:00 | 691.20 | 666.87 | 655.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 14:30:00 | 692.90 | 676.94 | 661.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-20 12:15:00 | 760.32 | 754.43 | 742.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 757.95 | 778.96 | 779.86 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 814.20 | 786.01 | 782.98 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 768.00 | 781.41 | 781.71 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 792.00 | 779.73 | 778.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 802.00 | 784.55 | 781.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 11:15:00 | 794.65 | 797.11 | 791.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:00:00 | 794.65 | 797.11 | 791.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 789.45 | 795.58 | 791.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:30:00 | 788.40 | 795.58 | 791.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 794.50 | 795.36 | 791.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 14:45:00 | 795.95 | 794.68 | 791.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 15:15:00 | 788.00 | 793.35 | 791.24 | SL hit (close<static) qty=1.00 sl=788.05 alert=retest2 |

### Cycle 40 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 768.85 | 788.45 | 789.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 758.70 | 782.50 | 786.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 725.15 | 724.67 | 736.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 15:00:00 | 725.15 | 724.67 | 736.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 675.35 | 651.98 | 657.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 675.35 | 651.98 | 657.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 675.35 | 656.65 | 659.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:30:00 | 674.65 | 656.65 | 659.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 673.00 | 662.07 | 661.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 674.00 | 664.45 | 662.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 662.85 | 667.13 | 664.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 662.85 | 667.13 | 664.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 662.85 | 667.13 | 664.50 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 658.90 | 662.43 | 662.73 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 15:15:00 | 665.00 | 663.10 | 662.96 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 660.90 | 662.66 | 662.78 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 665.50 | 663.30 | 663.02 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 660.00 | 662.55 | 662.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 11:15:00 | 653.40 | 660.12 | 661.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 13:15:00 | 652.50 | 649.80 | 653.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 13:15:00 | 652.50 | 649.80 | 653.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 652.50 | 649.80 | 653.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:00:00 | 652.50 | 649.80 | 653.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 673.55 | 654.55 | 655.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 673.55 | 654.55 | 655.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 662.50 | 656.14 | 656.15 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 676.15 | 660.14 | 657.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 09:15:00 | 693.35 | 675.74 | 668.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 750.15 | 766.95 | 753.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 750.15 | 766.95 | 753.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 750.15 | 766.95 | 753.64 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 735.00 | 748.99 | 749.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 725.30 | 744.25 | 747.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 13:15:00 | 743.95 | 743.53 | 745.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 14:00:00 | 743.95 | 743.53 | 745.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 722.45 | 739.31 | 743.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 15:00:00 | 722.45 | 739.31 | 743.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 738.75 | 731.07 | 735.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 738.75 | 731.07 | 735.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 737.60 | 732.38 | 735.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 733.30 | 732.38 | 735.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 742.60 | 734.42 | 736.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 742.60 | 734.42 | 736.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 741.40 | 735.82 | 736.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 742.15 | 735.82 | 736.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 12:15:00 | 742.80 | 738.09 | 737.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 15:15:00 | 743.60 | 740.14 | 738.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 762.05 | 762.25 | 756.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 762.05 | 762.25 | 756.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 762.05 | 762.25 | 756.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:15:00 | 763.45 | 762.25 | 756.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 758.50 | 761.50 | 756.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:15:00 | 742.55 | 761.50 | 756.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 732.15 | 755.63 | 754.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 732.15 | 755.63 | 754.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 730.00 | 750.50 | 752.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 721.45 | 744.69 | 749.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 14:15:00 | 701.75 | 694.08 | 706.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 15:00:00 | 701.75 | 694.08 | 706.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 707.05 | 696.84 | 705.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 707.05 | 696.84 | 705.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 711.95 | 699.86 | 706.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 711.95 | 699.86 | 706.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 720.10 | 703.91 | 707.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 720.10 | 703.91 | 707.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 13:15:00 | 722.20 | 710.60 | 710.17 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 700.00 | 711.40 | 711.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 685.00 | 701.01 | 706.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 09:15:00 | 713.70 | 694.39 | 698.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 09:15:00 | 713.70 | 694.39 | 698.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 713.70 | 694.39 | 698.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:30:00 | 717.75 | 694.39 | 698.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 704.25 | 696.36 | 699.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:30:00 | 713.70 | 696.36 | 699.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 697.00 | 696.48 | 698.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 14:00:00 | 693.85 | 695.95 | 698.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:15:00 | 693.50 | 696.69 | 698.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 15:15:00 | 696.35 | 685.18 | 684.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 15:15:00 | 696.35 | 685.18 | 684.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 11:15:00 | 709.40 | 692.86 | 688.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 15:15:00 | 711.90 | 712.39 | 705.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-27 09:15:00 | 748.90 | 712.39 | 705.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 726.90 | 732.38 | 721.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 09:15:00 | 733.10 | 732.38 | 721.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 718.55 | 729.61 | 721.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 718.55 | 729.61 | 721.58 | SL hit (close<ema400) qty=1.00 sl=721.58 alert=retest1 |

### Cycle 54 — SELL (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 09:15:00 | 705.80 | 725.69 | 728.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 15:15:00 | 691.75 | 704.58 | 714.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 715.30 | 706.72 | 714.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 715.30 | 706.72 | 714.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 715.30 | 706.72 | 714.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 11:45:00 | 704.30 | 707.82 | 714.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 15:15:00 | 705.00 | 707.73 | 712.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-06 13:15:00 | 704.95 | 708.24 | 711.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-06 14:00:00 | 705.35 | 707.66 | 710.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 14:15:00 | 669.08 | 682.36 | 690.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 14:15:00 | 669.75 | 682.36 | 690.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 14:15:00 | 669.70 | 682.36 | 690.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 14:15:00 | 670.08 | 682.36 | 690.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 653.00 | 651.12 | 659.35 | SL hit (close>ema200) qty=0.50 sl=651.12 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 12:15:00 | 664.65 | 658.33 | 657.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 14:15:00 | 668.25 | 661.16 | 659.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 14:15:00 | 670.00 | 673.48 | 667.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 14:45:00 | 673.95 | 673.48 | 667.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 675.10 | 673.88 | 669.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 10:15:00 | 685.40 | 673.88 | 669.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 698.00 | 711.82 | 713.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 698.00 | 711.82 | 713.22 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 728.95 | 709.69 | 708.22 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 697.50 | 711.36 | 712.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 687.65 | 698.27 | 700.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 15:15:00 | 689.95 | 689.93 | 694.28 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-07 09:15:00 | 641.15 | 689.93 | 694.28 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 609.09 | 683.70 | 691.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 689.30 | 674.38 | 680.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 689.30 | 674.38 | 680.78 | SL hit (close>ema200) qty=0.50 sl=674.38 alert=retest1 |

### Cycle 59 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 725.60 | 692.32 | 688.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 734.65 | 712.49 | 704.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 830.00 | 831.17 | 810.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 830.00 | 831.17 | 810.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 817.10 | 826.67 | 818.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 815.50 | 826.67 | 818.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 820.25 | 825.39 | 818.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:30:00 | 816.30 | 825.39 | 818.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 816.75 | 823.66 | 818.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:00:00 | 816.75 | 823.66 | 818.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 816.60 | 822.25 | 818.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 816.60 | 822.25 | 818.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 811.85 | 820.17 | 817.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:00:00 | 811.85 | 820.17 | 817.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 809.60 | 815.63 | 815.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 802.25 | 812.95 | 814.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 682.50 | 680.48 | 690.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 682.50 | 680.48 | 690.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 706.95 | 687.48 | 689.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:00:00 | 706.95 | 687.48 | 689.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 700.55 | 690.10 | 690.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 696.45 | 691.03 | 691.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:30:00 | 694.10 | 691.03 | 691.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 661.63 | 684.08 | 687.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 691.00 | 678.63 | 681.94 | SL hit (close>ema200) qty=0.50 sl=678.63 alert=retest2 |

### Cycle 61 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 693.75 | 684.11 | 684.01 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-05-14 09:15:00)

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

### Cycle 63 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 680.30 | 675.58 | 675.56 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 671.35 | 674.84 | 675.23 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-05-22 09:15:00)

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

### Cycle 66 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 688.15 | 698.86 | 700.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 684.15 | 694.82 | 698.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 686.45 | 684.77 | 690.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:15:00 | 685.85 | 684.77 | 690.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 681.80 | 684.17 | 689.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:30:00 | 680.10 | 684.72 | 689.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 696.45 | 687.07 | 690.14 | SL hit (close>static) qty=1.00 sl=691.60 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-30 14:15:00)

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

### Cycle 68 — SELL (started 2025-06-03 10:15:00)

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

### Cycle 69 — BUY (started 2025-06-06 09:15:00)

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

### Cycle 70 — SELL (started 2025-06-13 09:15:00)

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

### Cycle 71 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 768.00 | 761.62 | 761.26 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 758.50 | 760.94 | 761.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 756.25 | 760.00 | 760.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 750.10 | 748.82 | 753.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 750.10 | 748.82 | 753.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 750.10 | 748.82 | 753.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:45:00 | 734.60 | 746.38 | 751.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 15:15:00 | 745.95 | 740.03 | 739.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-24 15:15:00)

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

### Cycle 74 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 762.90 | 764.36 | 764.53 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-07-03 09:15:00)

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

### Cycle 76 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 766.45 | 770.18 | 770.32 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 776.15 | 771.37 | 770.85 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-07-08 10:15:00)

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

### Cycle 79 — BUY (started 2025-07-11 11:15:00)

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

### Cycle 80 — SELL (started 2025-07-15 12:15:00)

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

### Cycle 81 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 580.00 | 578.70 | 578.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 585.00 | 579.96 | 579.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 589.60 | 592.98 | 589.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 589.60 | 592.98 | 589.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 589.60 | 592.98 | 589.04 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-08-20 10:15:00)

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

### Cycle 83 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 584.30 | 583.81 | 583.78 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-25 12:15:00)

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

### Cycle 85 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 535.65 | 533.90 | 533.80 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 531.40 | 533.40 | 533.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 14:15:00 | 529.60 | 532.18 | 532.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 535.15 | 532.52 | 532.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 535.15 | 532.52 | 532.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 535.15 | 532.52 | 532.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 536.35 | 532.52 | 532.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 536.50 | 533.32 | 533.27 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 529.90 | 533.15 | 533.52 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-09-12 12:15:00)

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

### Cycle 90 — SELL (started 2025-09-18 11:15:00)

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

### Cycle 91 — BUY (started 2025-09-30 09:15:00)

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

### Cycle 92 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 530.60 | 533.58 | 533.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 526.45 | 532.15 | 533.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 522.40 | 522.19 | 525.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 522.40 | 522.19 | 525.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 525.65 | 523.55 | 525.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 524.90 | 523.55 | 525.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 523.25 | 523.49 | 525.36 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 529.10 | 526.86 | 526.57 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 525.15 | 526.41 | 526.41 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 528.05 | 526.73 | 526.56 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-13 12:15:00)

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

### Cycle 97 — BUY (started 2025-10-15 11:15:00)

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

### Cycle 98 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 525.40 | 527.41 | 527.42 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-17 14:15:00)

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

### Cycle 100 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 526.35 | 533.60 | 534.24 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 537.00 | 533.57 | 533.36 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 531.10 | 533.23 | 533.40 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-29 12:15:00)

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

### Cycle 104 — SELL (started 2025-11-06 11:15:00)

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

### Cycle 105 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 648.25 | 638.19 | 637.46 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-10 11:15:00)

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

### Cycle 107 — BUY (started 2025-11-13 12:15:00)

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

### Cycle 108 — SELL (started 2025-11-17 13:15:00)

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

### Cycle 109 — BUY (started 2025-11-21 15:15:00)

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

### Cycle 110 — SELL (started 2025-11-25 10:15:00)

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

### Cycle 111 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 592.50 | 584.35 | 583.58 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-05 11:15:00)

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

### Cycle 113 — BUY (started 2025-12-15 09:15:00)

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

### Cycle 114 — SELL (started 2025-12-19 12:15:00)

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

### Cycle 115 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 517.85 | 512.61 | 512.61 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 508.00 | 514.71 | 514.78 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 14:15:00 | 516.70 | 511.28 | 511.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 521.40 | 514.22 | 512.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 13:15:00 | 512.50 | 515.37 | 513.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 13:15:00 | 512.50 | 515.37 | 513.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 512.50 | 515.37 | 513.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 512.35 | 515.37 | 513.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 515.00 | 515.30 | 513.88 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 508.60 | 513.05 | 513.15 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 516.95 | 512.23 | 512.16 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-28 09:15:00)

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

### Cycle 121 — BUY (started 2026-02-04 11:15:00)

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

### Cycle 122 — SELL (started 2026-02-05 11:15:00)

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

### Cycle 123 — BUY (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 10:15:00 | 453.00 | 440.58 | 440.15 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 439.90 | 440.87 | 440.92 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-02-16 15:15:00)

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

### Cycle 126 — SELL (started 2026-02-18 15:15:00)

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

### Cycle 127 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 445.00 | 442.00 | 441.78 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-20 14:15:00)

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

### Cycle 129 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 373.40 | 359.82 | 359.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 375.60 | 362.98 | 360.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 365.00 | 369.86 | 365.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 365.00 | 369.86 | 365.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 365.00 | 369.86 | 365.84 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 354.85 | 362.87 | 363.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 352.20 | 357.75 | 360.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 349.00 | 348.97 | 353.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:00:00 | 349.00 | 348.97 | 353.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 356.55 | 350.14 | 353.01 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-03-24 12:15:00)

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

### Cycle 132 — SELL (started 2026-03-30 11:15:00)

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

### Cycle 133 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 366.10 | 361.06 | 360.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 376.00 | 367.43 | 364.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 15:15:00 | 423.20 | 426.56 | 415.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 09:15:00 | 412.60 | 426.56 | 415.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 412.75 | 423.80 | 415.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 418.65 | 423.80 | 415.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 10:15:00 | 460.51 | 447.71 | 436.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 485.05 | 489.71 | 490.22 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-24 14:15:00)

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

### Cycle 136 — SELL (started 2026-04-29 11:15:00)

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

### Cycle 137 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 489.00 | 483.24 | 483.09 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-05-07 09:15:00)

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
| SELL | retest2 | 2024-05-13 11:45:00 | 728.45 | 2024-05-14 14:15:00 | 744.25 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-05-23 10:30:00 | 719.55 | 2024-05-29 13:15:00 | 713.70 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2024-05-23 11:45:00 | 720.00 | 2024-05-29 13:15:00 | 713.70 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2024-05-23 13:30:00 | 719.50 | 2024-05-29 13:15:00 | 713.70 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2024-05-23 14:00:00 | 719.55 | 2024-05-29 13:15:00 | 713.70 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2024-05-31 09:15:00 | 721.40 | 2024-06-07 09:15:00 | 793.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-22 09:15:00 | 760.00 | 2024-07-25 09:15:00 | 722.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 10:15:00 | 760.50 | 2024-07-25 09:15:00 | 722.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 14:00:00 | 759.70 | 2024-07-25 09:15:00 | 721.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 09:15:00 | 760.00 | 2024-07-26 09:15:00 | 749.80 | STOP_HIT | 0.50 | 1.34% |
| SELL | retest2 | 2024-07-22 10:15:00 | 760.50 | 2024-07-26 09:15:00 | 749.80 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2024-07-22 14:00:00 | 759.70 | 2024-07-26 09:15:00 | 749.80 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2024-07-23 09:15:00 | 749.95 | 2024-07-26 14:15:00 | 757.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-08-09 10:15:00 | 719.30 | 2024-08-16 11:15:00 | 710.45 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-08-20 13:30:00 | 713.15 | 2024-09-06 09:15:00 | 750.00 | STOP_HIT | 1.00 | 5.17% |
| BUY | retest2 | 2024-08-21 10:30:00 | 716.50 | 2024-09-06 09:15:00 | 750.00 | STOP_HIT | 1.00 | 4.68% |
| SELL | retest2 | 2024-09-11 09:15:00 | 735.50 | 2024-09-13 12:15:00 | 749.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-09-11 10:15:00 | 739.05 | 2024-09-13 12:15:00 | 749.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-09-11 12:00:00 | 743.20 | 2024-09-13 12:15:00 | 749.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-09-11 13:30:00 | 743.60 | 2024-09-13 12:15:00 | 749.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-12 13:15:00 | 741.65 | 2024-09-13 12:15:00 | 749.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-09-13 10:00:00 | 741.55 | 2024-09-13 12:15:00 | 749.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-09-19 12:15:00 | 770.20 | 2024-09-24 14:15:00 | 771.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-09-19 13:15:00 | 769.80 | 2024-09-24 14:15:00 | 771.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2024-09-19 15:15:00 | 772.85 | 2024-09-24 14:15:00 | 771.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-09-20 10:00:00 | 777.25 | 2024-09-24 14:15:00 | 771.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-09-27 09:30:00 | 806.05 | 2024-09-30 09:15:00 | 785.15 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-10-14 11:00:00 | 812.05 | 2024-10-15 15:15:00 | 893.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-14 12:45:00 | 813.50 | 2024-10-15 15:15:00 | 894.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-14 14:00:00 | 812.55 | 2024-10-15 15:15:00 | 893.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-14 14:30:00 | 814.20 | 2024-10-15 15:15:00 | 895.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-18 09:15:00 | 631.80 | 2024-11-21 14:15:00 | 643.45 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-11-18 11:45:00 | 636.35 | 2024-11-25 10:15:00 | 640.45 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-11-18 13:30:00 | 636.20 | 2024-11-25 10:15:00 | 640.45 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-11-19 11:30:00 | 635.05 | 2024-11-25 10:15:00 | 640.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-11-19 15:15:00 | 632.05 | 2024-11-25 13:15:00 | 640.75 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-11-22 10:15:00 | 635.55 | 2024-11-25 13:15:00 | 640.75 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-11-22 11:00:00 | 635.55 | 2024-11-25 13:15:00 | 640.75 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-11-22 12:00:00 | 635.55 | 2024-11-25 13:15:00 | 640.75 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-12-16 12:45:00 | 691.20 | 2024-12-20 12:15:00 | 760.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-16 14:30:00 | 692.90 | 2024-12-20 12:15:00 | 762.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-03 14:45:00 | 795.95 | 2025-01-03 15:15:00 | 788.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-02-18 14:00:00 | 693.85 | 2025-02-21 15:15:00 | 696.35 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-02-18 15:15:00 | 693.50 | 2025-02-21 15:15:00 | 696.35 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-02-27 09:15:00 | 748.90 | 2025-02-28 09:15:00 | 718.55 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-02-28 10:45:00 | 736.60 | 2025-03-04 09:15:00 | 705.80 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2025-02-28 12:15:00 | 736.65 | 2025-03-04 09:15:00 | 705.80 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2025-02-28 13:00:00 | 739.85 | 2025-03-04 09:15:00 | 705.80 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest2 | 2025-02-28 14:45:00 | 737.40 | 2025-03-04 09:15:00 | 705.80 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2025-03-05 11:45:00 | 704.30 | 2025-03-10 14:15:00 | 669.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-05 15:15:00 | 705.00 | 2025-03-10 14:15:00 | 669.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-06 13:15:00 | 704.95 | 2025-03-10 14:15:00 | 669.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-06 14:00:00 | 705.35 | 2025-03-10 14:15:00 | 670.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-05 11:45:00 | 704.30 | 2025-03-13 09:15:00 | 653.00 | STOP_HIT | 0.50 | 7.28% |
| SELL | retest2 | 2025-03-05 15:15:00 | 705.00 | 2025-03-13 09:15:00 | 653.00 | STOP_HIT | 0.50 | 7.38% |
| SELL | retest2 | 2025-03-06 13:15:00 | 704.95 | 2025-03-13 09:15:00 | 653.00 | STOP_HIT | 0.50 | 7.37% |
| SELL | retest2 | 2025-03-06 14:00:00 | 705.35 | 2025-03-13 09:15:00 | 653.00 | STOP_HIT | 0.50 | 7.42% |
| BUY | retest2 | 2025-03-19 10:15:00 | 685.40 | 2025-03-25 12:15:00 | 698.00 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest1 | 2025-04-07 09:15:00 | 641.15 | 2025-04-07 09:15:00 | 609.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-04-07 09:15:00 | 641.15 | 2025-04-08 09:15:00 | 689.30 | STOP_HIT | 0.50 | -7.51% |
| SELL | retest2 | 2025-05-08 11:30:00 | 696.45 | 2025-05-09 09:15:00 | 661.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 11:30:00 | 696.45 | 2025-05-12 09:15:00 | 691.00 | STOP_HIT | 0.50 | 0.78% |
| SELL | retest2 | 2025-05-08 12:30:00 | 694.10 | 2025-05-12 11:15:00 | 693.75 | STOP_HIT | 1.00 | 0.05% |
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
