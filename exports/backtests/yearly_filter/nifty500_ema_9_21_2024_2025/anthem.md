# Anthem Biosciences Ltd. (ANTHEM)

## Backtest Summary

- **Window:** 2025-07-21 09:15:00 → 2026-05-11 15:15:00 (1381 bars)
- **Last close:** 785.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 58 |
| ALERT1 | 39 |
| ALERT2 | 39 |
| ALERT2_SKIP | 21 |
| ALERT3 | 103 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 39 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 13 / 27
- **Target hits / Stop hits / Partials:** 2 / 35 / 3
- **Avg / median % per leg:** 0.24% / -0.55%
- **Sum % (uncompounded):** 9.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 11 | 0 | 0.78% | 10.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 5 | 38.5% | 2 | 11 | 0 | 0.78% | 10.2% |
| SELL (all) | 27 | 8 | 29.6% | 0 | 24 | 3 | -0.02% | -0.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.75% | -1.8% |
| SELL @ 3rd Alert (retest2) | 26 | 8 | 30.8% | 0 | 23 | 3 | 0.05% | 1.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.75% | -1.8% |
| retest2 (combined) | 39 | 13 | 33.3% | 2 | 34 | 3 | 0.29% | 11.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 749.50 | 762.23 | 763.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 739.55 | 751.14 | 756.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 751.50 | 747.03 | 752.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 751.50 | 747.03 | 752.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 751.50 | 747.03 | 752.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 751.30 | 747.03 | 752.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 752.55 | 748.13 | 752.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 752.55 | 748.13 | 752.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 749.60 | 748.43 | 752.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:30:00 | 746.90 | 748.57 | 750.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 745.65 | 735.79 | 737.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 750.00 | 740.72 | 739.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 750.00 | 740.72 | 739.59 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 727.10 | 740.09 | 740.19 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 755.00 | 742.33 | 740.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 755.70 | 749.83 | 745.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 747.90 | 749.44 | 745.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:00:00 | 747.90 | 749.44 | 745.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 745.95 | 748.74 | 745.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 745.95 | 748.74 | 745.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 743.00 | 747.59 | 745.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 750.80 | 747.59 | 745.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-14 14:15:00 | 825.88 | 797.98 | 777.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 835.25 | 838.60 | 838.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 829.35 | 836.75 | 837.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 832.05 | 825.93 | 830.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 832.05 | 825.93 | 830.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 832.05 | 825.93 | 830.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 831.85 | 825.93 | 830.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 834.35 | 827.61 | 830.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:15:00 | 833.45 | 827.61 | 830.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 823.00 | 824.31 | 827.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 822.25 | 824.31 | 827.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 812.55 | 810.29 | 814.47 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 824.80 | 817.10 | 816.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 842.95 | 822.27 | 819.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 829.80 | 836.24 | 829.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 829.80 | 836.24 | 829.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 829.80 | 836.24 | 829.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 829.30 | 836.24 | 829.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 834.50 | 835.89 | 830.09 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 823.50 | 829.48 | 829.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 816.70 | 826.92 | 828.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 11:15:00 | 824.10 | 823.17 | 825.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:45:00 | 825.00 | 823.17 | 825.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 827.80 | 823.78 | 825.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 827.30 | 823.78 | 825.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 822.70 | 823.56 | 824.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 820.80 | 823.56 | 824.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 822.25 | 821.04 | 822.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 839.35 | 824.79 | 823.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 839.35 | 824.79 | 823.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 846.50 | 834.55 | 830.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 831.15 | 835.62 | 832.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 12:15:00 | 831.15 | 835.62 | 832.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 831.15 | 835.62 | 832.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 831.15 | 835.62 | 832.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 833.00 | 835.10 | 832.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 833.00 | 835.10 | 832.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 831.90 | 834.46 | 832.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 831.90 | 834.46 | 832.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 836.00 | 834.77 | 832.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 844.45 | 834.77 | 832.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 840.50 | 837.31 | 834.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 846.05 | 848.80 | 848.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 846.05 | 848.80 | 848.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 838.45 | 843.60 | 846.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 14:15:00 | 821.45 | 820.57 | 826.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 15:00:00 | 821.45 | 820.57 | 826.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 771.00 | 764.89 | 769.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 771.00 | 764.89 | 769.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 771.50 | 766.21 | 770.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 762.40 | 766.21 | 770.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 763.90 | 757.46 | 760.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 763.90 | 757.46 | 760.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 764.45 | 758.86 | 761.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 767.50 | 758.86 | 761.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 775.25 | 763.54 | 762.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 776.80 | 768.12 | 765.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 774.45 | 775.43 | 771.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 774.45 | 775.43 | 771.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 774.45 | 775.43 | 771.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 774.20 | 775.43 | 771.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 766.35 | 773.61 | 771.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 775.20 | 773.61 | 771.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 770.60 | 773.01 | 771.38 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 764.30 | 770.99 | 771.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 10:15:00 | 761.80 | 769.15 | 770.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 12:15:00 | 780.80 | 770.82 | 770.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 12:15:00 | 780.80 | 770.82 | 770.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 780.80 | 770.82 | 770.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:30:00 | 774.90 | 770.82 | 770.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 792.30 | 775.11 | 772.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 800.00 | 780.09 | 775.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 782.05 | 784.05 | 778.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:15:00 | 781.70 | 784.05 | 778.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 778.95 | 782.53 | 778.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:45:00 | 777.25 | 782.53 | 778.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 784.05 | 782.83 | 779.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 777.50 | 782.83 | 779.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 789.60 | 784.76 | 781.15 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 779.00 | 780.27 | 780.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 764.15 | 773.07 | 775.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 743.80 | 738.65 | 749.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 745.50 | 738.65 | 749.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 728.50 | 737.34 | 747.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 724.90 | 735.62 | 745.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:30:00 | 725.55 | 724.40 | 732.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:30:00 | 721.65 | 722.77 | 725.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 726.15 | 720.19 | 721.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 13:15:00 | 725.00 | 722.23 | 722.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 725.00 | 722.23 | 722.15 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 719.50 | 721.98 | 722.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 14:15:00 | 716.95 | 719.94 | 721.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 10:15:00 | 719.75 | 718.97 | 720.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 719.75 | 718.97 | 720.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 719.75 | 718.97 | 720.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 719.80 | 718.97 | 720.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 708.80 | 705.41 | 709.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 708.80 | 705.41 | 709.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 693.75 | 701.96 | 706.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 692.05 | 701.96 | 706.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 692.55 | 698.67 | 703.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:00:00 | 692.80 | 697.50 | 702.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 688.80 | 697.13 | 701.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 698.75 | 696.90 | 699.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:30:00 | 698.70 | 696.90 | 699.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 698.00 | 697.12 | 699.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 698.10 | 697.12 | 699.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 679.20 | 693.53 | 697.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:30:00 | 677.40 | 689.46 | 695.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 705.00 | 690.71 | 694.63 | SL hit (close>static) qty=1.00 sl=703.95 alert=retest2 |

### Cycle 16 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 705.00 | 697.71 | 697.21 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 679.80 | 694.13 | 695.63 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 706.00 | 689.37 | 689.04 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 683.95 | 692.48 | 693.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 681.55 | 690.30 | 691.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 647.75 | 645.58 | 654.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:00:00 | 647.75 | 645.58 | 654.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 648.45 | 646.16 | 653.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 658.25 | 646.16 | 653.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 636.10 | 633.49 | 636.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 629.25 | 632.64 | 634.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 636.30 | 632.17 | 631.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 636.30 | 632.17 | 631.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 661.10 | 638.91 | 634.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 652.55 | 653.88 | 646.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 652.55 | 653.88 | 646.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 652.55 | 653.88 | 646.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 651.75 | 653.88 | 646.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 646.95 | 651.23 | 648.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 642.75 | 651.23 | 648.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 648.75 | 650.74 | 648.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 650.65 | 650.74 | 648.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 654.30 | 655.62 | 655.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 13:15:00 | 654.30 | 655.62 | 655.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 647.25 | 652.81 | 654.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 650.00 | 649.89 | 651.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:15:00 | 641.90 | 649.89 | 651.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 653.15 | 648.65 | 650.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 653.15 | 648.65 | 650.69 | SL hit (close>ema400) qty=1.00 sl=650.69 alert=retest1 |

### Cycle 22 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 660.35 | 652.41 | 652.13 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 648.10 | 652.18 | 652.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 643.60 | 650.46 | 651.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 651.95 | 648.96 | 650.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 651.95 | 648.96 | 650.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 651.95 | 648.96 | 650.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 651.90 | 648.96 | 650.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 661.80 | 651.53 | 651.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 669.50 | 655.13 | 653.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 678.40 | 692.01 | 684.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 678.40 | 692.01 | 684.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 678.40 | 692.01 | 684.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 678.40 | 692.01 | 684.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 673.35 | 688.28 | 683.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 673.10 | 688.28 | 683.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 675.00 | 680.22 | 680.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 670.50 | 678.27 | 679.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 13:15:00 | 664.65 | 664.58 | 669.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:45:00 | 664.35 | 664.58 | 669.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 678.30 | 667.60 | 669.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 681.90 | 670.40 | 670.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 680.00 | 672.32 | 671.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 686.40 | 679.46 | 676.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 681.00 | 682.09 | 678.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 681.00 | 682.09 | 678.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 675.90 | 680.66 | 678.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 673.20 | 680.66 | 678.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 674.05 | 679.34 | 678.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 674.05 | 679.34 | 678.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 671.65 | 677.40 | 677.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 15:15:00 | 667.00 | 673.68 | 675.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 643.80 | 640.90 | 646.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:15:00 | 643.80 | 640.90 | 646.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 646.35 | 642.43 | 646.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 645.25 | 642.43 | 646.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 648.95 | 643.73 | 646.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 648.95 | 643.73 | 646.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 649.65 | 644.92 | 646.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 649.65 | 644.92 | 646.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 654.15 | 648.41 | 648.06 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 639.75 | 647.79 | 647.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 638.95 | 646.03 | 647.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 651.30 | 643.72 | 645.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 651.30 | 643.72 | 645.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 651.30 | 643.72 | 645.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 651.30 | 643.72 | 645.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 652.20 | 645.42 | 645.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 648.65 | 645.42 | 645.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 649.85 | 646.30 | 646.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 649.85 | 646.30 | 646.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 655.45 | 649.12 | 647.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 650.95 | 651.32 | 649.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:45:00 | 650.85 | 651.32 | 649.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 650.05 | 651.07 | 649.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 650.05 | 651.07 | 649.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 650.45 | 650.94 | 649.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:15:00 | 645.05 | 650.94 | 649.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 645.05 | 649.77 | 649.14 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 647.15 | 648.70 | 648.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 642.55 | 647.16 | 648.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 655.80 | 648.34 | 648.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 655.80 | 648.34 | 648.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 655.80 | 648.34 | 648.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 655.80 | 648.34 | 648.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 656.50 | 649.97 | 649.11 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 640.60 | 649.01 | 649.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 633.80 | 645.97 | 648.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 619.50 | 619.13 | 627.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 614.35 | 619.13 | 627.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 625.05 | 619.85 | 626.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 625.05 | 619.85 | 626.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 624.30 | 620.74 | 625.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:30:00 | 625.50 | 620.74 | 625.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 624.45 | 621.48 | 625.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:45:00 | 624.00 | 621.48 | 625.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 626.30 | 622.44 | 625.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 626.30 | 622.44 | 625.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 629.65 | 623.89 | 626.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 629.65 | 623.89 | 626.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 628.00 | 624.71 | 626.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 624.45 | 624.73 | 626.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 623.60 | 624.57 | 625.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 623.95 | 624.57 | 625.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 603.90 | 615.08 | 617.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 593.23 | 606.89 | 612.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 592.42 | 606.89 | 612.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 592.75 | 606.89 | 612.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 607.50 | 597.82 | 601.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 607.50 | 597.82 | 601.96 | SL hit (close>ema200) qty=0.50 sl=597.82 alert=retest2 |

### Cycle 34 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 604.90 | 603.99 | 603.89 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 596.40 | 604.91 | 605.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 587.30 | 595.35 | 599.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 599.50 | 595.05 | 598.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 599.50 | 595.05 | 598.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 599.50 | 595.05 | 598.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 599.60 | 595.05 | 598.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 598.35 | 595.71 | 598.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:30:00 | 598.85 | 595.71 | 598.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 598.00 | 596.17 | 598.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 592.75 | 595.65 | 597.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 596.00 | 592.64 | 594.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:45:00 | 595.75 | 593.62 | 594.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 12:15:00 | 595.50 | 593.62 | 594.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 599.00 | 595.08 | 595.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 599.00 | 595.08 | 595.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 625.05 | 602.16 | 598.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 599.25 | 601.58 | 598.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 599.25 | 601.58 | 598.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 599.25 | 601.58 | 598.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 599.25 | 601.58 | 598.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 590.10 | 599.28 | 597.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:45:00 | 590.00 | 599.28 | 597.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 590.00 | 597.43 | 597.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 594.75 | 597.43 | 597.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 587.10 | 595.36 | 596.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 587.10 | 595.36 | 596.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 585.00 | 591.52 | 594.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 592.90 | 590.80 | 592.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 592.90 | 590.80 | 592.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 592.90 | 590.80 | 592.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 592.90 | 590.80 | 592.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 594.50 | 591.54 | 592.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 602.80 | 591.54 | 592.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 598.35 | 592.90 | 593.47 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 611.40 | 596.60 | 595.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 13:15:00 | 630.20 | 607.22 | 604.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 689.00 | 697.28 | 688.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 689.00 | 697.28 | 688.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 689.00 | 697.28 | 688.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 690.90 | 697.28 | 688.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 687.75 | 695.38 | 688.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:15:00 | 687.20 | 695.38 | 688.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 690.55 | 694.41 | 688.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:30:00 | 692.85 | 689.37 | 688.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 692.15 | 692.27 | 690.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:15:00 | 692.35 | 692.02 | 690.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 685.90 | 690.29 | 690.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 14:15:00 | 685.90 | 690.29 | 690.44 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 696.80 | 691.53 | 690.85 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 685.00 | 691.62 | 691.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 682.30 | 689.16 | 690.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 691.65 | 687.72 | 689.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 691.65 | 687.72 | 689.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 691.65 | 687.72 | 689.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 691.65 | 687.72 | 689.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 695.15 | 689.20 | 690.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 694.10 | 689.20 | 690.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 695.50 | 690.46 | 690.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:30:00 | 693.35 | 690.46 | 690.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 695.65 | 691.50 | 691.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 696.80 | 693.16 | 691.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 10:15:00 | 691.65 | 693.62 | 692.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 10:15:00 | 691.65 | 693.62 | 692.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 691.65 | 693.62 | 692.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 691.65 | 693.62 | 692.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 692.25 | 693.34 | 692.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:45:00 | 690.30 | 693.34 | 692.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 691.55 | 692.98 | 692.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:45:00 | 691.50 | 692.98 | 692.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 693.85 | 693.16 | 692.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 691.35 | 693.16 | 692.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 694.50 | 693.43 | 692.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 694.50 | 693.43 | 692.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 700.00 | 694.74 | 693.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 703.05 | 697.34 | 694.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 702.75 | 698.24 | 695.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:00:00 | 702.40 | 699.08 | 695.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 15:00:00 | 702.90 | 701.45 | 697.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 704.75 | 702.76 | 699.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 690.25 | 698.03 | 698.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 690.25 | 698.03 | 698.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 635.00 | 658.20 | 666.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 13:15:00 | 650.55 | 648.80 | 658.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:00:00 | 650.55 | 648.80 | 658.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 660.10 | 651.06 | 658.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 660.10 | 651.06 | 658.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 660.00 | 652.85 | 658.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 674.90 | 652.85 | 658.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 671.00 | 662.59 | 662.21 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 645.50 | 663.26 | 663.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 636.15 | 648.94 | 654.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 12:15:00 | 642.50 | 640.98 | 645.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 13:00:00 | 642.50 | 640.98 | 645.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 637.55 | 639.78 | 643.55 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 647.25 | 641.18 | 640.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 649.50 | 643.13 | 641.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 640.15 | 644.15 | 642.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 640.15 | 644.15 | 642.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 640.15 | 644.15 | 642.90 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 639.50 | 642.12 | 642.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 634.30 | 640.55 | 641.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 15:15:00 | 639.00 | 638.44 | 640.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:15:00 | 641.05 | 638.44 | 640.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 648.10 | 640.37 | 640.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 648.10 | 640.37 | 640.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 649.20 | 642.13 | 641.70 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 633.50 | 642.31 | 642.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 629.05 | 639.66 | 641.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 631.85 | 631.13 | 634.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 631.85 | 631.13 | 634.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 636.45 | 632.42 | 634.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 640.75 | 632.42 | 634.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 639.35 | 633.80 | 635.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 639.35 | 633.80 | 635.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 646.65 | 636.37 | 636.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 656.60 | 640.42 | 638.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 641.30 | 644.25 | 641.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 641.30 | 644.25 | 641.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 641.30 | 644.25 | 641.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:45:00 | 657.70 | 648.42 | 644.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 11:15:00 | 723.47 | 703.74 | 688.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 13:15:00 | 722.85 | 725.17 | 725.37 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 728.80 | 725.49 | 725.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 13:15:00 | 732.90 | 727.28 | 726.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 14:15:00 | 724.30 | 726.68 | 726.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 14:15:00 | 724.30 | 726.68 | 726.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 724.30 | 726.68 | 726.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 15:00:00 | 724.30 | 726.68 | 726.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 724.90 | 726.32 | 725.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 727.80 | 726.32 | 725.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 10:15:00 | 722.30 | 725.19 | 725.46 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 749.00 | 728.61 | 726.68 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 741.65 | 743.47 | 743.60 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 757.55 | 744.26 | 743.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 768.20 | 749.05 | 745.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 762.00 | 762.60 | 755.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 10:00:00 | 762.00 | 762.60 | 755.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 775.75 | 764.95 | 757.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 759.85 | 764.95 | 757.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 766.20 | 771.15 | 767.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:15:00 | 766.65 | 771.15 | 767.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 766.60 | 770.24 | 767.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 768.80 | 770.24 | 767.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 761.15 | 768.42 | 767.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:45:00 | 760.20 | 768.42 | 767.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 766.10 | 767.96 | 767.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 761.55 | 767.96 | 767.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 761.75 | 766.72 | 766.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 761.75 | 766.72 | 766.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 14:15:00 | 759.65 | 765.30 | 765.98 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 772.80 | 765.46 | 764.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 780.15 | 770.22 | 767.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 12:15:00 | 770.60 | 771.37 | 768.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 13:00:00 | 770.60 | 771.37 | 768.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 775.00 | 781.86 | 778.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-11 12:00:00 | 784.20 | 782.07 | 779.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-11 14:45:00 | 784.80 | 782.61 | 780.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-11 15:15:00 | 785.75 | 782.61 | 780.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-05 12:30:00 | 746.90 | 2025-08-07 15:15:00 | 750.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-08-07 14:00:00 | 745.65 | 2025-08-07 15:15:00 | 750.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-08-13 09:15:00 | 750.80 | 2025-08-14 14:15:00 | 825.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-10 11:15:00 | 820.80 | 2025-09-12 10:15:00 | 839.35 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-09-11 11:00:00 | 822.25 | 2025-09-12 10:15:00 | 839.35 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-09-17 09:15:00 | 844.45 | 2025-09-22 09:15:00 | 846.05 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-09-17 11:15:00 | 840.50 | 2025-09-22 09:15:00 | 846.05 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2025-10-23 11:15:00 | 724.90 | 2025-10-29 13:15:00 | 725.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-10-24 12:30:00 | 725.55 | 2025-10-29 13:15:00 | 725.00 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-10-28 09:30:00 | 721.65 | 2025-10-29 13:15:00 | 725.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-10-29 12:15:00 | 726.15 | 2025-10-29 13:15:00 | 725.00 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-11-06 10:15:00 | 692.05 | 2025-11-10 12:15:00 | 705.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-11-06 11:30:00 | 692.55 | 2025-11-10 15:15:00 | 705.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-06 13:00:00 | 692.80 | 2025-11-10 15:15:00 | 705.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-11-07 09:15:00 | 688.80 | 2025-11-10 15:15:00 | 705.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-11-10 10:30:00 | 677.40 | 2025-11-10 15:15:00 | 705.00 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-11-27 09:15:00 | 629.25 | 2025-11-28 13:15:00 | 636.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-12-03 11:15:00 | 650.65 | 2025-12-05 13:15:00 | 654.30 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest1 | 2025-12-09 09:15:00 | 641.90 | 2025-12-09 11:15:00 | 653.15 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-01-02 12:15:00 | 648.65 | 2026-01-02 12:15:00 | 649.85 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-01-14 09:30:00 | 624.45 | 2026-01-20 12:15:00 | 593.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:30:00 | 623.60 | 2026-01-20 12:15:00 | 592.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 623.95 | 2026-01-20 12:15:00 | 592.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 624.45 | 2026-01-22 09:15:00 | 607.50 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2026-01-14 10:30:00 | 623.60 | 2026-01-22 09:15:00 | 607.50 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2026-01-14 11:00:00 | 623.95 | 2026-01-22 09:15:00 | 607.50 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2026-01-20 09:15:00 | 603.90 | 2026-01-22 15:15:00 | 604.90 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2026-01-29 09:30:00 | 592.75 | 2026-01-30 13:15:00 | 599.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-30 11:00:00 | 596.00 | 2026-01-30 13:15:00 | 599.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-01-30 11:45:00 | 595.75 | 2026-01-30 13:15:00 | 599.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-01-30 12:15:00 | 595.50 | 2026-01-30 13:15:00 | 599.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-02-01 14:15:00 | 594.75 | 2026-02-01 14:15:00 | 587.10 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-02-17 09:30:00 | 692.85 | 2026-02-18 14:15:00 | 685.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-18 09:45:00 | 692.15 | 2026-02-18 14:15:00 | 685.90 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-02-18 11:15:00 | 692.35 | 2026-02-18 14:15:00 | 685.90 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-25 09:30:00 | 703.05 | 2026-03-02 09:15:00 | 690.25 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-02-25 11:15:00 | 702.75 | 2026-03-02 09:15:00 | 690.25 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-02-25 12:00:00 | 702.40 | 2026-03-02 09:15:00 | 690.25 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-02-25 15:00:00 | 702.90 | 2026-03-02 09:15:00 | 690.25 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-03-27 13:45:00 | 657.70 | 2026-04-09 11:15:00 | 723.47 | TARGET_HIT | 1.00 | 10.00% |
