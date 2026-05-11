# Chalet Hotels Ltd. (CHALET)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 787.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 155 |
| ALERT1 | 105 |
| ALERT2 | 105 |
| ALERT2_SKIP | 58 |
| ALERT3 | 273 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 118 |
| PARTIAL | 12 |
| TARGET_HIT | 5 |
| STOP_HIT | 120 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 91
- **Target hits / Stop hits / Partials:** 5 / 120 / 12
- **Avg / median % per leg:** 0.27% / -0.78%
- **Sum % (uncompounded):** 37.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 15 | 25.0% | 4 | 55 | 1 | 0.02% | 1.2% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.36% | 1.8% |
| BUY @ 3rd Alert (retest2) | 55 | 13 | 23.6% | 4 | 51 | 0 | -0.01% | -0.6% |
| SELL (all) | 77 | 31 | 40.3% | 1 | 65 | 11 | 0.47% | 36.5% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.50% | -4.5% |
| SELL @ 3rd Alert (retest2) | 74 | 31 | 41.9% | 1 | 62 | 11 | 0.55% | 41.0% |
| retest1 (combined) | 8 | 2 | 25.0% | 0 | 7 | 1 | -0.34% | -2.7% |
| retest2 (combined) | 129 | 44 | 34.1% | 5 | 113 | 11 | 0.31% | 40.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 795.25 | 786.76 | 786.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 800.10 | 789.43 | 787.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 782.85 | 788.52 | 787.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 782.85 | 788.52 | 787.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 782.85 | 788.52 | 787.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:30:00 | 781.65 | 788.52 | 787.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 774.50 | 785.71 | 786.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 11:15:00 | 771.85 | 782.94 | 785.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 769.00 | 768.77 | 774.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 14:30:00 | 772.60 | 768.77 | 774.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 776.40 | 769.71 | 773.61 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 781.65 | 776.05 | 775.57 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 772.85 | 775.37 | 775.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 12:15:00 | 764.80 | 772.61 | 774.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 14:15:00 | 771.15 | 771.15 | 773.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 15:00:00 | 771.15 | 771.15 | 773.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 773.90 | 771.70 | 773.03 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 783.80 | 775.25 | 774.42 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 767.00 | 773.72 | 774.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 764.55 | 771.88 | 773.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 775.20 | 768.30 | 770.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 12:15:00 | 775.20 | 768.30 | 770.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 775.20 | 768.30 | 770.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:30:00 | 771.00 | 768.30 | 770.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 768.00 | 768.24 | 770.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 773.50 | 768.24 | 770.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 756.65 | 765.30 | 768.46 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 792.40 | 769.64 | 767.82 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 724.95 | 771.11 | 773.89 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 785.95 | 767.61 | 765.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 12:15:00 | 788.00 | 776.97 | 770.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 14:15:00 | 775.30 | 778.02 | 772.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 15:00:00 | 775.30 | 778.02 | 772.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 838.00 | 844.76 | 837.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 850.05 | 844.76 | 837.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 10:30:00 | 847.95 | 844.21 | 838.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 845.20 | 843.91 | 839.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 13:15:00 | 833.20 | 841.54 | 838.84 | SL hit (close<static) qty=1.00 sl=835.40 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 15:15:00 | 831.95 | 836.93 | 837.05 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 850.95 | 839.73 | 838.31 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 832.85 | 837.57 | 837.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 14:15:00 | 826.40 | 835.33 | 836.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 809.40 | 808.34 | 815.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 13:15:00 | 822.80 | 811.49 | 814.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 822.80 | 811.49 | 814.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:00:00 | 822.80 | 811.49 | 814.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 822.05 | 813.60 | 815.60 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 825.15 | 817.70 | 817.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 14:15:00 | 836.05 | 822.95 | 820.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 15:15:00 | 837.00 | 840.03 | 832.98 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:15:00 | 844.25 | 840.03 | 832.98 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 836.70 | 838.58 | 833.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:45:00 | 835.35 | 838.58 | 833.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 833.95 | 837.65 | 833.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:00:00 | 833.95 | 837.65 | 833.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 833.00 | 836.72 | 833.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 833.00 | 836.72 | 833.49 | SL hit (close<ema400) qty=1.00 sl=833.49 alert=retest1 |

### Cycle 14 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 822.70 | 830.82 | 831.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 813.30 | 823.42 | 827.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 828.25 | 823.84 | 826.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 828.25 | 823.84 | 826.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 828.25 | 823.84 | 826.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 829.20 | 823.84 | 826.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 816.40 | 822.35 | 825.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 11:15:00 | 807.55 | 822.35 | 825.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:30:00 | 810.75 | 817.51 | 822.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:00:00 | 813.05 | 814.32 | 819.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:45:00 | 812.95 | 813.74 | 818.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 805.15 | 807.55 | 813.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 841.95 | 818.10 | 816.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 841.95 | 818.10 | 816.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 857.25 | 825.93 | 820.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 844.90 | 855.94 | 843.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 14:15:00 | 844.90 | 855.94 | 843.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 844.90 | 855.94 | 843.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 15:00:00 | 844.90 | 855.94 | 843.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 848.60 | 854.47 | 843.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 860.10 | 854.47 | 843.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:00:00 | 851.40 | 851.77 | 849.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:45:00 | 850.50 | 852.20 | 849.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:15:00 | 853.25 | 851.65 | 850.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 853.20 | 851.96 | 850.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 09:15:00 | 861.00 | 851.63 | 851.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:00:00 | 858.70 | 853.04 | 851.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 12:15:00 | 846.10 | 851.12 | 851.12 | SL hit (close<static) qty=1.00 sl=848.80 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 846.90 | 850.28 | 850.74 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 854.35 | 851.07 | 850.82 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 844.55 | 849.53 | 850.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 15:15:00 | 841.10 | 846.60 | 848.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 843.05 | 840.69 | 843.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 843.05 | 840.69 | 843.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 843.05 | 840.69 | 843.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 840.70 | 840.69 | 843.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:15:00 | 798.66 | 816.46 | 826.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-19 09:15:00 | 756.63 | 770.91 | 783.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 815.65 | 780.55 | 780.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 15:15:00 | 820.50 | 799.48 | 790.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 11:15:00 | 799.95 | 800.49 | 793.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 11:30:00 | 799.30 | 800.49 | 793.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 801.70 | 800.73 | 793.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 809.00 | 803.47 | 796.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:45:00 | 809.95 | 804.58 | 798.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 10:45:00 | 809.00 | 805.00 | 798.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 13:15:00 | 812.10 | 805.20 | 799.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 829.95 | 827.87 | 820.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 14:00:00 | 834.75 | 831.07 | 824.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 855.50 | 832.07 | 829.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 12:15:00 | 841.45 | 843.02 | 843.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 841.45 | 843.02 | 843.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 833.95 | 840.83 | 842.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 808.45 | 807.33 | 817.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:00:00 | 790.55 | 802.38 | 811.86 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 799.55 | 792.41 | 801.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 801.25 | 792.41 | 801.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 800.05 | 793.94 | 800.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 801.40 | 793.94 | 800.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 802.65 | 795.68 | 801.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 14:15:00 | 802.65 | 795.68 | 801.12 | SL hit (close>ema400) qty=1.00 sl=801.12 alert=retest1 |

### Cycle 21 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 776.00 | 772.94 | 772.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 788.15 | 775.98 | 774.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 12:15:00 | 813.65 | 815.70 | 806.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 13:00:00 | 813.65 | 815.70 | 806.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 812.00 | 814.96 | 807.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:30:00 | 810.95 | 814.96 | 807.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 815.00 | 816.28 | 811.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 13:30:00 | 816.10 | 816.28 | 811.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 806.10 | 814.24 | 811.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 806.10 | 814.24 | 811.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 818.00 | 815.00 | 812.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 832.00 | 815.00 | 812.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 827.95 | 817.59 | 813.45 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 09:15:00 | 808.95 | 813.97 | 814.04 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 818.25 | 814.82 | 814.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 827.95 | 818.14 | 816.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 15:15:00 | 873.60 | 876.60 | 866.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 09:15:00 | 911.35 | 876.60 | 866.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 884.75 | 890.87 | 885.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 15:00:00 | 884.75 | 890.87 | 885.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 885.00 | 889.69 | 885.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 899.30 | 889.69 | 885.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 13:15:00 | 881.15 | 888.95 | 887.04 | SL hit (close<static) qty=1.00 sl=882.40 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 883.65 | 889.96 | 890.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 10:15:00 | 873.25 | 885.30 | 888.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 890.00 | 884.55 | 887.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 12:15:00 | 890.00 | 884.55 | 887.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 890.00 | 884.55 | 887.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 890.00 | 884.55 | 887.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 892.45 | 886.13 | 887.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:30:00 | 892.00 | 886.13 | 887.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 895.20 | 889.40 | 889.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 10:15:00 | 905.55 | 892.38 | 890.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 13:15:00 | 891.10 | 894.94 | 892.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 13:15:00 | 891.10 | 894.94 | 892.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 891.10 | 894.94 | 892.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:00:00 | 891.10 | 894.94 | 892.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 889.00 | 893.75 | 892.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:15:00 | 895.00 | 893.75 | 892.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 895.00 | 894.00 | 892.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 902.60 | 894.00 | 892.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 13:15:00 | 881.40 | 889.71 | 890.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 881.40 | 889.71 | 890.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 875.35 | 880.95 | 884.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 12:15:00 | 863.35 | 860.20 | 868.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 13:00:00 | 863.35 | 860.20 | 868.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 874.40 | 863.04 | 869.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:45:00 | 874.95 | 863.04 | 869.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 877.60 | 865.95 | 869.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 877.60 | 865.95 | 869.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 880.50 | 868.86 | 870.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 874.60 | 868.86 | 870.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 879.50 | 872.52 | 872.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 889.05 | 877.14 | 874.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 879.45 | 883.63 | 880.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 879.45 | 883.63 | 880.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 879.45 | 883.63 | 880.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 879.45 | 883.63 | 880.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 891.65 | 885.24 | 881.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 11:15:00 | 902.95 | 885.24 | 881.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 868.00 | 897.42 | 897.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 868.00 | 897.42 | 897.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 15:15:00 | 863.00 | 870.54 | 875.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 872.95 | 871.02 | 875.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:45:00 | 870.35 | 871.02 | 875.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 878.00 | 871.95 | 874.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 878.00 | 871.95 | 874.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 876.35 | 872.83 | 874.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:30:00 | 870.00 | 871.45 | 874.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:00:00 | 865.95 | 871.45 | 874.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 15:15:00 | 826.50 | 845.39 | 857.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 822.65 | 845.32 | 856.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 865.45 | 849.35 | 857.03 | SL hit (close>ema200) qty=0.50 sl=849.35 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 886.80 | 863.03 | 861.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 895.25 | 872.19 | 866.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 889.10 | 889.81 | 882.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 889.10 | 889.81 | 882.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 885.00 | 888.85 | 882.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 885.00 | 888.85 | 882.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 885.00 | 887.45 | 883.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 875.65 | 884.98 | 882.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 877.15 | 883.41 | 881.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:15:00 | 881.60 | 882.51 | 881.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 13:15:00 | 886.90 | 890.06 | 890.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 886.90 | 890.06 | 890.13 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 15:15:00 | 895.00 | 890.52 | 890.29 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 876.50 | 887.72 | 889.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 867.20 | 880.06 | 884.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 870.85 | 866.81 | 874.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 870.85 | 866.81 | 874.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 870.70 | 868.08 | 873.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 870.70 | 868.08 | 873.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 872.40 | 868.94 | 873.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 868.20 | 868.94 | 873.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 869.90 | 868.42 | 871.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 11:15:00 | 824.79 | 843.95 | 852.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 11:15:00 | 826.40 | 843.95 | 852.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 15:15:00 | 841.65 | 841.61 | 848.79 | SL hit (close>ema200) qty=0.50 sl=841.61 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 12:15:00 | 880.60 | 852.78 | 850.32 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 830.65 | 847.86 | 849.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 10:15:00 | 822.20 | 842.73 | 846.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 15:15:00 | 860.05 | 837.14 | 841.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 15:15:00 | 860.05 | 837.14 | 841.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 860.05 | 837.14 | 841.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:45:00 | 821.80 | 833.49 | 838.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 14:15:00 | 837.65 | 836.38 | 836.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 837.65 | 836.38 | 836.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 15:15:00 | 844.65 | 838.04 | 837.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 13:15:00 | 863.90 | 865.91 | 853.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 14:00:00 | 863.90 | 865.91 | 853.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 864.10 | 874.45 | 863.00 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 849.15 | 860.14 | 861.27 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 870.25 | 861.90 | 861.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 874.20 | 864.36 | 862.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 871.45 | 871.71 | 868.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 10:15:00 | 871.45 | 871.71 | 868.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 871.45 | 871.71 | 868.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 869.05 | 871.71 | 868.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 870.10 | 871.39 | 868.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 870.10 | 871.39 | 868.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 890.90 | 889.64 | 882.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:30:00 | 920.45 | 897.31 | 886.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 15:15:00 | 879.00 | 892.37 | 889.12 | SL hit (close<static) qty=1.00 sl=882.25 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 876.00 | 886.84 | 887.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 866.50 | 882.77 | 885.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 839.55 | 834.97 | 851.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 839.55 | 834.97 | 851.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 840.00 | 839.10 | 846.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 831.95 | 839.23 | 845.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 838.25 | 838.94 | 844.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 11:15:00 | 851.80 | 846.34 | 845.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 851.80 | 846.34 | 845.71 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 842.05 | 845.35 | 845.60 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 885.75 | 850.11 | 846.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 903.35 | 860.75 | 851.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 882.80 | 885.16 | 871.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 11:15:00 | 878.85 | 885.16 | 871.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 880.00 | 885.71 | 880.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 879.95 | 885.71 | 880.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 878.10 | 884.19 | 880.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 878.10 | 884.19 | 880.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 888.40 | 883.92 | 881.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 886.05 | 883.92 | 881.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 890.50 | 896.40 | 891.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 892.40 | 896.40 | 891.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 890.05 | 895.13 | 891.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:15:00 | 890.95 | 895.13 | 891.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 890.00 | 894.10 | 890.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:15:00 | 888.00 | 894.10 | 890.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 888.00 | 892.88 | 890.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 894.05 | 892.49 | 890.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 10:15:00 | 881.10 | 890.21 | 889.81 | SL hit (close<static) qty=1.00 sl=885.50 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 881.90 | 888.55 | 889.09 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 14:15:00 | 890.00 | 887.09 | 886.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 901.70 | 890.46 | 888.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 892.00 | 892.99 | 890.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 15:15:00 | 892.00 | 892.99 | 890.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 892.00 | 892.99 | 890.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 899.10 | 892.99 | 890.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 889.00 | 892.19 | 890.68 | SL hit (close<static) qty=1.00 sl=890.20 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 978.05 | 995.91 | 996.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 941.20 | 978.42 | 987.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 13:15:00 | 970.95 | 967.14 | 977.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 13:30:00 | 968.10 | 967.14 | 977.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 966.10 | 965.65 | 973.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:15:00 | 961.05 | 966.21 | 973.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 979.80 | 966.81 | 971.53 | SL hit (close>static) qty=1.00 sl=976.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 975.75 | 973.25 | 973.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 980.65 | 975.17 | 974.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 11:15:00 | 972.45 | 975.39 | 974.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 11:15:00 | 972.45 | 975.39 | 974.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 972.45 | 975.39 | 974.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:00:00 | 972.45 | 975.39 | 974.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 973.95 | 975.10 | 974.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 987.75 | 975.10 | 974.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 09:45:00 | 983.45 | 982.06 | 980.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 957.65 | 977.18 | 978.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 957.65 | 977.18 | 978.57 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 982.60 | 976.92 | 976.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 1012.65 | 986.40 | 981.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 997.15 | 1005.97 | 998.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 997.15 | 1005.97 | 998.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 997.15 | 1005.97 | 998.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 997.15 | 1005.97 | 998.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 998.40 | 1004.46 | 998.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:30:00 | 996.15 | 1004.46 | 998.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 992.70 | 1002.11 | 998.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 992.70 | 1002.11 | 998.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 991.75 | 1000.04 | 997.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 14:45:00 | 996.25 | 996.69 | 996.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 15:15:00 | 1000.00 | 996.69 | 996.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 985.40 | 994.96 | 995.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 985.40 | 994.96 | 995.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 968.45 | 989.66 | 993.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 14:15:00 | 810.75 | 809.30 | 819.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 14:30:00 | 811.45 | 809.30 | 819.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 788.15 | 783.03 | 789.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 788.00 | 783.03 | 789.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 788.70 | 784.16 | 789.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:30:00 | 783.25 | 784.65 | 788.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 744.09 | 759.02 | 770.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 738.50 | 732.07 | 744.77 | SL hit (close>ema200) qty=0.50 sl=732.07 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 756.05 | 746.35 | 746.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 770.00 | 757.31 | 753.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 762.55 | 765.75 | 760.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 10:15:00 | 762.55 | 765.75 | 760.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 762.55 | 765.75 | 760.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 762.55 | 765.75 | 760.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 759.65 | 764.53 | 760.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 758.00 | 764.53 | 760.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 761.95 | 764.01 | 760.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 764.40 | 763.72 | 760.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-03 11:15:00 | 840.84 | 796.01 | 778.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 783.85 | 791.01 | 791.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 15:15:00 | 772.00 | 787.21 | 789.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 11:15:00 | 757.05 | 755.14 | 766.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 12:00:00 | 757.05 | 755.14 | 766.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 706.95 | 721.75 | 736.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:15:00 | 704.80 | 716.69 | 731.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 669.56 | 681.29 | 691.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-14 11:15:00 | 684.60 | 681.08 | 689.52 | SL hit (close>ema200) qty=0.50 sl=681.08 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 697.35 | 679.18 | 677.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 701.50 | 688.40 | 682.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 701.00 | 704.51 | 696.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 701.00 | 704.51 | 696.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 715.85 | 706.78 | 698.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 722.70 | 709.12 | 700.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:15:00 | 717.75 | 711.69 | 702.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:45:00 | 719.65 | 713.15 | 708.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 11:15:00 | 702.95 | 718.27 | 719.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 11:15:00 | 702.95 | 718.27 | 719.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 697.85 | 714.19 | 717.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 733.10 | 715.90 | 717.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 733.10 | 715.90 | 717.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 733.10 | 715.90 | 717.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 733.10 | 715.90 | 717.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 754.25 | 723.57 | 720.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 767.95 | 753.94 | 743.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 10:15:00 | 772.55 | 773.80 | 764.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:15:00 | 780.65 | 773.80 | 764.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 14:15:00 | 819.68 | 795.10 | 784.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 791.10 | 795.10 | 784.66 | SL hit (close<ema400) qty=0.50 sl=795.10 alert=retest1 |

### Cycle 54 — SELL (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 15:15:00 | 772.15 | 780.15 | 780.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 762.75 | 776.67 | 779.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 780.25 | 777.39 | 779.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 780.25 | 777.39 | 779.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 780.25 | 777.39 | 779.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 780.25 | 777.39 | 779.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 769.80 | 775.87 | 778.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:45:00 | 781.50 | 775.87 | 778.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 780.20 | 775.96 | 778.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:00:00 | 780.20 | 775.96 | 778.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 781.60 | 777.09 | 778.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 765.50 | 777.73 | 778.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 10:30:00 | 775.95 | 769.20 | 770.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 799.15 | 775.26 | 772.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 799.15 | 775.26 | 772.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 801.05 | 791.09 | 783.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 15:15:00 | 862.00 | 862.25 | 847.72 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 09:15:00 | 867.90 | 862.25 | 847.72 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 853.20 | 859.79 | 852.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:30:00 | 854.05 | 859.79 | 852.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 844.60 | 856.75 | 851.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 844.60 | 856.75 | 851.36 | SL hit (close<ema400) qty=1.00 sl=851.36 alert=retest1 |

### Cycle 56 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 838.55 | 848.20 | 849.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 827.30 | 840.37 | 844.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 836.65 | 832.02 | 838.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-01 10:00:00 | 836.65 | 832.02 | 838.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 828.85 | 831.38 | 837.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:15:00 | 825.45 | 831.30 | 837.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 840.55 | 832.62 | 836.22 | SL hit (close>static) qty=1.00 sl=839.85 alert=retest2 |

### Cycle 57 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 858.55 | 841.39 | 839.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 866.15 | 855.05 | 847.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 845.75 | 861.18 | 856.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 845.75 | 861.18 | 856.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 845.75 | 861.18 | 856.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 845.75 | 861.18 | 856.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 841.35 | 857.21 | 854.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 839.85 | 857.21 | 854.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 836.15 | 853.00 | 853.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 832.00 | 848.80 | 851.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 815.00 | 808.76 | 824.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 815.00 | 808.76 | 824.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 822.10 | 811.66 | 822.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 806.50 | 821.04 | 823.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 810.55 | 818.94 | 822.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 15:15:00 | 825.00 | 817.12 | 816.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 825.00 | 817.12 | 816.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 830.55 | 819.81 | 817.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 825.15 | 827.73 | 823.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 825.15 | 827.73 | 823.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 825.15 | 827.73 | 823.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 09:30:00 | 823.00 | 827.73 | 823.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 823.60 | 826.90 | 823.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 11:45:00 | 827.00 | 827.03 | 824.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 13:45:00 | 826.10 | 826.78 | 824.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 11:15:00 | 821.90 | 823.67 | 823.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 11:15:00 | 821.90 | 823.67 | 823.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 13:15:00 | 819.15 | 822.58 | 823.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 09:15:00 | 823.50 | 819.44 | 821.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 823.50 | 819.44 | 821.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 823.50 | 819.44 | 821.30 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 832.15 | 822.92 | 822.50 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 822.80 | 824.27 | 824.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 14:15:00 | 819.25 | 822.79 | 823.70 | Break + close below crossover candle low |

### Cycle 63 — BUY (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 09:15:00 | 830.95 | 824.36 | 824.25 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 814.00 | 822.53 | 823.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 801.55 | 813.01 | 816.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 804.80 | 801.68 | 807.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 804.80 | 801.68 | 807.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 804.80 | 801.68 | 807.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 804.80 | 801.68 | 807.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 800.00 | 801.88 | 806.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:30:00 | 800.90 | 801.88 | 806.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 806.20 | 802.75 | 806.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 15:00:00 | 806.20 | 802.75 | 806.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 800.35 | 802.27 | 805.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 814.70 | 802.27 | 805.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 826.10 | 807.03 | 807.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 826.10 | 807.03 | 807.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 828.05 | 811.24 | 809.31 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 810.05 | 814.21 | 814.58 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 819.90 | 815.35 | 815.06 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 11:15:00 | 810.10 | 814.30 | 814.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 13:15:00 | 808.95 | 812.50 | 813.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 812.75 | 812.55 | 813.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 812.75 | 812.55 | 813.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 812.75 | 812.55 | 813.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 812.75 | 812.55 | 813.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 819.30 | 814.04 | 814.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:15:00 | 819.15 | 814.04 | 814.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 826.05 | 816.45 | 815.20 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 801.80 | 813.39 | 814.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 781.15 | 804.41 | 809.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 793.70 | 793.23 | 801.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 13:30:00 | 796.70 | 793.23 | 801.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 798.45 | 794.27 | 801.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 798.45 | 794.27 | 801.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 799.00 | 795.22 | 801.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 829.00 | 795.22 | 801.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 836.75 | 803.52 | 804.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 836.75 | 803.52 | 804.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 847.05 | 812.23 | 808.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 864.30 | 822.64 | 813.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 15:15:00 | 910.10 | 914.78 | 898.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 09:15:00 | 907.35 | 914.78 | 898.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 901.70 | 909.88 | 899.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 14:00:00 | 903.55 | 906.52 | 900.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:00:00 | 903.85 | 905.98 | 900.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 908.05 | 905.92 | 901.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 894.70 | 906.00 | 907.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 894.70 | 906.00 | 907.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 888.30 | 897.46 | 902.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 885.95 | 885.23 | 891.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 885.95 | 885.23 | 891.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 885.95 | 885.23 | 891.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 885.95 | 885.23 | 891.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 888.10 | 885.81 | 890.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 893.95 | 886.30 | 890.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 892.30 | 887.50 | 890.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 892.00 | 887.50 | 890.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 894.15 | 888.83 | 891.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 893.85 | 888.83 | 891.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 887.85 | 888.64 | 890.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:15:00 | 887.25 | 888.64 | 890.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:00:00 | 886.85 | 887.88 | 890.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 898.95 | 889.99 | 890.65 | SL hit (close>static) qty=1.00 sl=897.30 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 912.90 | 894.57 | 892.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 915.15 | 910.82 | 904.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 10:15:00 | 909.05 | 914.48 | 909.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 909.05 | 914.48 | 909.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 909.05 | 914.48 | 909.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 909.05 | 914.48 | 909.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 919.60 | 915.50 | 910.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:15:00 | 921.80 | 915.50 | 910.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 920.00 | 917.26 | 912.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 10:00:00 | 921.15 | 926.00 | 921.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 928.00 | 919.62 | 919.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 922.35 | 924.07 | 922.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 922.35 | 924.07 | 922.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 917.70 | 922.79 | 922.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 916.45 | 920.79 | 921.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 916.45 | 920.79 | 921.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 914.90 | 919.61 | 920.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 921.15 | 917.38 | 919.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 921.15 | 917.38 | 919.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 921.15 | 917.38 | 919.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 923.95 | 917.38 | 919.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 919.15 | 917.74 | 919.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:30:00 | 915.80 | 917.19 | 918.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 12:00:00 | 915.00 | 917.19 | 918.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 915.40 | 916.80 | 918.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 09:45:00 | 915.40 | 917.61 | 918.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 914.25 | 916.93 | 917.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 919.45 | 916.93 | 917.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 916.70 | 916.25 | 917.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:00:00 | 916.70 | 916.25 | 917.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 919.90 | 916.98 | 917.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:30:00 | 920.50 | 916.98 | 917.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 918.00 | 917.18 | 917.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 915.30 | 917.18 | 917.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 921.00 | 917.95 | 917.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 922.50 | 918.86 | 918.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 922.50 | 918.86 | 918.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 934.00 | 921.89 | 919.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 12:15:00 | 922.00 | 922.68 | 920.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 13:00:00 | 922.00 | 922.68 | 920.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 921.05 | 922.72 | 920.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 921.05 | 922.72 | 920.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 920.00 | 922.17 | 920.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 925.55 | 922.17 | 920.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 12:15:00 | 912.05 | 921.10 | 922.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 912.05 | 921.10 | 922.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 15:15:00 | 911.00 | 916.74 | 919.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 916.95 | 916.78 | 919.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 916.95 | 916.78 | 919.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 916.95 | 916.78 | 919.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 919.15 | 916.78 | 919.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 914.90 | 911.11 | 915.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 914.90 | 911.11 | 915.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 909.95 | 910.87 | 914.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 906.50 | 910.87 | 914.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 905.45 | 909.79 | 913.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:45:00 | 898.05 | 905.17 | 910.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 13:15:00 | 895.75 | 890.88 | 890.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 895.75 | 890.88 | 890.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 902.10 | 894.11 | 892.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 893.40 | 899.05 | 896.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 893.40 | 899.05 | 896.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 893.40 | 899.05 | 896.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 891.55 | 899.05 | 896.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 892.45 | 897.73 | 896.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:30:00 | 890.85 | 897.73 | 896.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 893.70 | 895.17 | 895.19 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 15:15:00 | 897.80 | 895.22 | 895.18 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 889.30 | 894.04 | 894.65 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 903.00 | 892.49 | 892.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 910.35 | 896.06 | 894.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 897.65 | 898.48 | 895.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 897.65 | 898.48 | 895.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 897.65 | 898.48 | 895.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 895.90 | 898.48 | 895.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 898.00 | 899.36 | 896.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:45:00 | 908.50 | 901.14 | 897.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 906.90 | 913.31 | 913.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 906.90 | 913.31 | 913.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 903.30 | 911.31 | 912.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 892.80 | 890.99 | 896.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 10:15:00 | 892.80 | 890.99 | 896.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 892.80 | 890.99 | 896.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 896.70 | 890.99 | 896.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 892.75 | 888.83 | 892.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:15:00 | 897.75 | 888.83 | 892.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 894.35 | 889.93 | 892.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 891.20 | 889.93 | 892.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 891.55 | 890.26 | 892.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 885.00 | 890.26 | 892.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 881.30 | 873.06 | 872.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 881.30 | 873.06 | 872.84 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 870.00 | 872.60 | 872.68 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 874.00 | 872.88 | 872.80 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 870.15 | 872.33 | 872.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 860.35 | 869.56 | 871.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 863.35 | 863.26 | 866.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 863.35 | 863.26 | 866.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 865.50 | 863.37 | 865.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 865.50 | 863.37 | 865.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 875.00 | 865.69 | 866.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 880.10 | 865.69 | 866.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 891.45 | 870.84 | 868.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 896.85 | 876.05 | 871.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 891.55 | 891.97 | 884.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:00:00 | 891.55 | 891.97 | 884.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 919.00 | 926.47 | 923.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 919.00 | 926.47 | 923.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 920.00 | 925.18 | 922.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:30:00 | 919.85 | 925.18 | 922.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 920.00 | 924.09 | 922.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 905.10 | 924.09 | 922.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 915.10 | 922.29 | 921.98 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 916.80 | 921.19 | 921.51 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 927.80 | 922.13 | 921.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 15:15:00 | 937.65 | 926.72 | 924.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 933.00 | 937.18 | 932.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 933.00 | 937.18 | 932.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 933.00 | 937.18 | 932.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 930.80 | 937.18 | 932.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 931.80 | 936.10 | 932.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 930.10 | 936.10 | 932.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 929.00 | 934.68 | 932.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 929.00 | 934.68 | 932.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 924.40 | 932.11 | 931.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:45:00 | 925.00 | 932.11 | 931.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 920.00 | 929.69 | 930.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 914.75 | 923.62 | 927.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 12:15:00 | 902.10 | 898.21 | 904.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 12:15:00 | 902.10 | 898.21 | 904.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 902.10 | 898.21 | 904.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:30:00 | 905.25 | 898.21 | 904.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 904.30 | 899.04 | 903.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 904.50 | 899.04 | 903.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 911.05 | 901.45 | 903.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 911.05 | 901.45 | 903.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 910.25 | 903.21 | 904.46 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 914.00 | 905.36 | 905.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 15:15:00 | 917.00 | 909.42 | 907.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 13:15:00 | 912.80 | 932.50 | 922.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 13:15:00 | 912.80 | 932.50 | 922.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 912.80 | 932.50 | 922.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 912.80 | 932.50 | 922.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 893.30 | 924.66 | 920.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 893.30 | 924.66 | 920.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 882.00 | 916.12 | 916.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 858.10 | 873.16 | 887.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 869.40 | 868.23 | 880.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 15:00:00 | 869.40 | 868.23 | 880.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 877.75 | 869.95 | 879.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 872.25 | 869.95 | 879.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 874.85 | 870.93 | 879.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 874.85 | 870.93 | 879.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 880.10 | 872.77 | 879.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:45:00 | 880.55 | 872.77 | 879.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 876.00 | 873.41 | 878.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 873.85 | 873.41 | 878.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:00:00 | 874.80 | 875.70 | 878.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:00:00 | 873.20 | 876.04 | 877.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:45:00 | 871.00 | 872.38 | 875.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 874.90 | 872.88 | 875.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 874.90 | 872.88 | 875.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 874.80 | 873.26 | 875.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 875.95 | 873.26 | 875.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 875.40 | 873.69 | 875.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 875.60 | 873.69 | 875.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 870.55 | 873.06 | 874.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 880.30 | 875.80 | 875.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 880.30 | 875.80 | 875.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 887.30 | 878.68 | 876.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 957.00 | 957.85 | 941.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:45:00 | 954.30 | 957.85 | 941.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1013.35 | 1025.82 | 1018.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 1011.00 | 1025.82 | 1018.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1012.45 | 1023.15 | 1017.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1011.00 | 1023.15 | 1017.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1021.45 | 1022.81 | 1018.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:15:00 | 1023.75 | 1022.81 | 1018.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 1003.45 | 1015.14 | 1015.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 1003.45 | 1015.14 | 1015.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 999.55 | 1012.02 | 1014.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 1025.95 | 1013.00 | 1014.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 11:15:00 | 1025.95 | 1013.00 | 1014.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1025.95 | 1013.00 | 1014.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 1027.60 | 1013.00 | 1014.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 1021.05 | 1014.61 | 1014.84 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 13:15:00 | 1025.30 | 1016.75 | 1015.79 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 1006.80 | 1013.86 | 1014.79 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 1019.70 | 1015.40 | 1014.95 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 994.70 | 1011.33 | 1013.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 10:15:00 | 988.50 | 1006.76 | 1010.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 1007.10 | 1006.02 | 1009.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 1007.10 | 1006.02 | 1009.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 1007.10 | 1006.02 | 1009.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 1007.10 | 1006.02 | 1009.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1027.50 | 1009.21 | 1009.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 1032.20 | 1009.21 | 1009.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1031.10 | 1013.59 | 1011.81 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 1010.20 | 1019.47 | 1019.70 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 1024.20 | 1020.42 | 1020.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 13:15:00 | 1027.00 | 1022.37 | 1021.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 1013.60 | 1022.53 | 1021.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1013.60 | 1022.53 | 1021.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1013.60 | 1022.53 | 1021.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 1001.60 | 1022.53 | 1021.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 1013.00 | 1020.63 | 1020.88 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 1024.80 | 1021.46 | 1021.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 12:15:00 | 1035.00 | 1024.17 | 1022.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 1027.70 | 1028.36 | 1025.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1036.90 | 1028.36 | 1025.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1035.70 | 1037.69 | 1033.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 1031.80 | 1037.69 | 1033.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1031.50 | 1036.45 | 1032.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 1031.50 | 1036.45 | 1032.96 | SL hit (close<ema400) qty=1.00 sl=1032.96 alert=retest1 |

### Cycle 104 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 1022.50 | 1030.36 | 1030.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 1020.30 | 1028.35 | 1029.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 10:15:00 | 1028.30 | 1026.45 | 1028.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 1028.30 | 1026.45 | 1028.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1028.30 | 1026.45 | 1028.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 1027.60 | 1026.45 | 1028.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 1029.30 | 1027.02 | 1028.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:45:00 | 1028.30 | 1027.02 | 1028.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 1027.10 | 1027.04 | 1028.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 1028.30 | 1027.04 | 1028.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1031.50 | 1025.69 | 1027.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 1031.50 | 1025.69 | 1027.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1031.70 | 1026.89 | 1027.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 1032.00 | 1026.89 | 1027.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 1034.50 | 1028.94 | 1028.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 1044.50 | 1033.17 | 1030.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 1053.80 | 1054.84 | 1045.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:45:00 | 1051.00 | 1054.84 | 1045.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1053.80 | 1057.32 | 1053.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 1052.50 | 1057.32 | 1053.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1055.40 | 1056.94 | 1053.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 1055.40 | 1056.94 | 1053.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1053.50 | 1056.25 | 1053.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 1053.80 | 1056.25 | 1053.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1061.50 | 1057.30 | 1054.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 1045.40 | 1057.30 | 1054.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1047.90 | 1055.42 | 1053.61 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 1032.20 | 1049.17 | 1050.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 1030.20 | 1045.38 | 1049.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 1017.40 | 1017.29 | 1026.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:30:00 | 1020.80 | 1017.29 | 1026.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1025.50 | 1020.17 | 1026.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1020.70 | 1021.13 | 1026.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:45:00 | 1021.00 | 1021.77 | 1025.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:00:00 | 1022.50 | 1023.09 | 1025.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 969.66 | 1000.47 | 1010.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 969.95 | 1000.47 | 1010.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 971.38 | 1000.47 | 1010.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 967.80 | 962.17 | 973.61 | SL hit (close>ema200) qty=0.50 sl=962.17 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 967.65 | 960.05 | 959.80 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 956.95 | 961.79 | 962.01 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 965.00 | 962.43 | 962.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 971.70 | 964.70 | 963.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 959.95 | 963.97 | 963.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 959.95 | 963.97 | 963.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 959.95 | 963.97 | 963.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 959.95 | 963.97 | 963.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 958.60 | 962.90 | 962.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 958.60 | 962.90 | 962.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 956.10 | 961.54 | 962.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 946.30 | 956.32 | 959.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 15:15:00 | 956.70 | 956.39 | 959.25 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 11:45:00 | 933.45 | 947.78 | 954.34 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 13:30:00 | 939.35 | 945.18 | 951.96 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 950.30 | 944.91 | 950.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 950.30 | 944.91 | 950.02 | SL hit (close>ema400) qty=1.00 sl=950.02 alert=retest1 |

### Cycle 111 — BUY (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 15:15:00 | 959.80 | 953.60 | 953.00 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 934.80 | 949.84 | 951.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 933.00 | 946.47 | 949.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 924.60 | 917.64 | 926.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 924.60 | 917.64 | 926.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 922.20 | 918.55 | 926.51 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 964.65 | 934.34 | 931.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 970.00 | 956.15 | 944.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 961.95 | 971.60 | 962.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 961.95 | 971.60 | 962.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 961.95 | 971.60 | 962.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 961.95 | 971.60 | 962.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 960.05 | 969.29 | 962.06 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 13:15:00 | 954.00 | 959.20 | 959.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 949.00 | 957.16 | 958.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 11:15:00 | 955.00 | 953.20 | 955.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 11:15:00 | 955.00 | 953.20 | 955.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 955.00 | 953.20 | 955.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 955.00 | 953.20 | 955.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 952.15 | 952.99 | 955.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:30:00 | 952.00 | 953.15 | 955.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 950.80 | 953.15 | 955.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:00:00 | 952.10 | 952.57 | 954.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:15:00 | 948.20 | 952.77 | 954.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 948.70 | 951.96 | 953.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:00:00 | 942.70 | 950.11 | 952.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 942.50 | 944.66 | 949.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 944.50 | 942.34 | 945.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 951.20 | 941.58 | 940.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 951.20 | 941.58 | 940.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 963.80 | 948.55 | 944.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 950.00 | 962.59 | 957.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 950.00 | 962.59 | 957.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 950.00 | 962.59 | 957.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:45:00 | 953.55 | 962.59 | 957.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 950.00 | 960.07 | 956.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 966.80 | 960.07 | 956.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 967.20 | 963.60 | 959.79 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 930.30 | 954.51 | 957.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 909.05 | 929.29 | 940.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 900.00 | 897.11 | 905.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 900.00 | 897.11 | 905.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 900.00 | 897.11 | 905.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 890.10 | 897.48 | 901.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 12:15:00 | 885.70 | 879.75 | 879.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 885.70 | 879.75 | 879.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 888.40 | 881.48 | 880.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 890.80 | 893.75 | 889.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 890.80 | 893.75 | 889.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 890.80 | 893.75 | 889.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 892.30 | 893.75 | 889.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 887.35 | 892.47 | 889.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 886.85 | 892.47 | 889.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 896.50 | 893.28 | 889.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:45:00 | 888.00 | 893.28 | 889.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 883.80 | 894.01 | 891.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 883.80 | 894.01 | 891.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 885.90 | 892.39 | 891.26 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 881.10 | 890.13 | 890.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 12:15:00 | 880.85 | 888.27 | 889.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 882.60 | 881.98 | 884.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 882.60 | 881.98 | 884.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 878.85 | 881.35 | 883.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 881.05 | 881.35 | 883.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 880.55 | 881.19 | 883.64 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 891.85 | 883.47 | 882.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 14:15:00 | 895.55 | 887.79 | 885.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 10:15:00 | 886.30 | 890.21 | 887.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 10:15:00 | 886.30 | 890.21 | 887.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 886.30 | 890.21 | 887.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 886.30 | 890.21 | 887.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 885.65 | 889.30 | 887.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:30:00 | 886.10 | 889.30 | 887.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 892.15 | 889.87 | 887.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:30:00 | 888.25 | 889.87 | 887.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 892.00 | 890.30 | 887.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 11:00:00 | 897.00 | 890.78 | 888.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 884.80 | 894.07 | 891.99 | SL hit (close<static) qty=1.00 sl=886.60 alert=retest2 |

### Cycle 120 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 880.00 | 889.70 | 890.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 878.20 | 885.77 | 888.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 905.25 | 887.82 | 888.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 905.25 | 887.82 | 888.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 905.25 | 887.82 | 888.42 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 10:15:00 | 905.05 | 891.26 | 889.93 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 12:15:00 | 890.50 | 896.61 | 897.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 883.40 | 890.79 | 893.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 884.20 | 882.09 | 886.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 884.20 | 882.09 | 886.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 884.20 | 882.09 | 886.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 884.00 | 882.09 | 886.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 890.40 | 883.75 | 886.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 891.35 | 883.75 | 886.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 890.40 | 885.08 | 887.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:30:00 | 887.90 | 886.51 | 887.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 883.90 | 879.76 | 882.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 892.00 | 882.41 | 883.20 | SL hit (close>static) qty=1.00 sl=891.90 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 893.85 | 884.70 | 884.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 899.25 | 889.26 | 886.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 885.90 | 892.61 | 889.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 885.90 | 892.61 | 889.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 885.90 | 892.61 | 889.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 885.90 | 892.61 | 889.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 884.10 | 890.91 | 888.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 884.05 | 890.91 | 888.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 885.00 | 889.88 | 889.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:15:00 | 877.50 | 889.88 | 889.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 876.55 | 887.22 | 887.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 874.40 | 884.65 | 886.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 881.25 | 880.21 | 883.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 881.25 | 880.21 | 883.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 881.25 | 880.21 | 883.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 884.00 | 880.21 | 883.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 882.30 | 880.63 | 883.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 881.70 | 880.63 | 883.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 871.90 | 878.89 | 882.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 871.70 | 878.89 | 882.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 872.60 | 868.74 | 868.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 872.60 | 868.74 | 868.57 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 11:15:00 | 866.65 | 868.33 | 868.42 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 869.75 | 868.62 | 868.54 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 14:15:00 | 867.10 | 868.48 | 868.50 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 869.55 | 868.69 | 868.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 896.30 | 874.21 | 871.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 880.85 | 884.02 | 877.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:00:00 | 880.85 | 884.02 | 877.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 885.95 | 884.41 | 878.37 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 866.60 | 876.15 | 877.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 858.45 | 866.45 | 870.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 873.45 | 866.44 | 869.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 873.45 | 866.44 | 869.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 873.45 | 866.44 | 869.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 873.45 | 866.44 | 869.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 872.20 | 867.59 | 869.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 865.25 | 867.59 | 869.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 868.50 | 865.39 | 866.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 874.95 | 867.31 | 867.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 874.95 | 867.31 | 867.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 890.00 | 873.81 | 870.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 896.40 | 896.41 | 886.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 12:00:00 | 896.40 | 896.41 | 886.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 891.80 | 897.25 | 890.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 890.50 | 897.25 | 890.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 887.90 | 895.38 | 890.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:15:00 | 881.95 | 895.38 | 890.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 879.35 | 892.17 | 889.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 879.45 | 892.17 | 889.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 874.90 | 886.72 | 887.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 872.45 | 882.01 | 885.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 876.40 | 873.54 | 877.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 10:00:00 | 876.40 | 873.54 | 877.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 888.75 | 876.58 | 878.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 890.00 | 876.58 | 878.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 886.70 | 878.61 | 879.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 887.95 | 878.61 | 879.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 12:15:00 | 887.50 | 880.38 | 880.20 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 873.50 | 879.39 | 879.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 868.05 | 877.12 | 878.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 14:15:00 | 876.15 | 874.95 | 877.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 14:15:00 | 876.15 | 874.95 | 877.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 876.15 | 874.95 | 877.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 876.15 | 874.95 | 877.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 869.05 | 873.77 | 876.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 866.15 | 873.77 | 876.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 878.10 | 874.64 | 876.66 | SL hit (close>static) qty=1.00 sl=876.60 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 873.10 | 870.62 | 870.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 884.75 | 878.58 | 874.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 878.05 | 882.31 | 879.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 878.05 | 882.31 | 879.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 878.05 | 882.31 | 879.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 878.05 | 882.31 | 879.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 871.30 | 880.11 | 878.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 866.15 | 880.11 | 878.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 864.70 | 877.03 | 877.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 847.00 | 865.60 | 870.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 851.85 | 843.04 | 851.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 851.85 | 843.04 | 851.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 851.85 | 843.04 | 851.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:15:00 | 850.05 | 843.04 | 851.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 850.00 | 844.43 | 851.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 849.40 | 844.24 | 850.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 806.93 | 825.21 | 834.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 821.90 | 818.72 | 826.22 | SL hit (close>ema200) qty=0.50 sl=818.72 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 838.20 | 827.84 | 827.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 854.40 | 834.39 | 830.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 860.10 | 860.66 | 851.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:30:00 | 863.00 | 860.66 | 851.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 835.75 | 853.91 | 850.84 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 838.10 | 847.87 | 848.45 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 859.95 | 850.44 | 849.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 870.70 | 854.49 | 851.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 868.65 | 880.57 | 870.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 868.65 | 880.57 | 870.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 868.65 | 880.57 | 870.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:45:00 | 868.15 | 880.57 | 870.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 869.85 | 878.43 | 870.34 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 859.60 | 866.83 | 867.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 855.80 | 864.63 | 866.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 855.45 | 850.70 | 855.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 855.45 | 850.70 | 855.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 855.45 | 850.70 | 855.52 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 864.85 | 857.66 | 857.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 868.05 | 862.64 | 860.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 868.65 | 875.64 | 869.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 868.65 | 875.64 | 869.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 868.65 | 875.64 | 869.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 871.15 | 875.64 | 869.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 870.30 | 874.57 | 869.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:00:00 | 877.40 | 873.04 | 870.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:00:00 | 880.80 | 875.75 | 872.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 857.20 | 874.53 | 874.09 | SL hit (close<static) qty=1.00 sl=868.10 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 855.20 | 870.67 | 872.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 846.00 | 858.85 | 864.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 853.85 | 853.56 | 860.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 853.85 | 853.56 | 860.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 862.00 | 855.24 | 860.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 862.00 | 855.24 | 860.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 865.20 | 857.24 | 861.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 865.20 | 857.24 | 861.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 857.80 | 857.35 | 860.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 852.50 | 857.35 | 860.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:00:00 | 855.80 | 857.04 | 860.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 868.65 | 859.36 | 861.09 | SL hit (close>static) qty=1.00 sl=865.20 alert=retest2 |

### Cycle 143 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 881.60 | 863.81 | 862.96 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 854.20 | 866.46 | 867.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 10:15:00 | 849.15 | 858.09 | 861.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 828.35 | 828.19 | 840.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 828.35 | 828.19 | 840.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 770.90 | 763.86 | 772.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 770.90 | 763.86 | 772.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 766.20 | 764.33 | 772.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:30:00 | 762.85 | 764.81 | 771.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:00:00 | 762.90 | 766.06 | 769.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 739.55 | 765.85 | 769.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 769.55 | 751.86 | 750.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 769.55 | 751.86 | 750.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 14:15:00 | 775.00 | 761.90 | 755.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 754.85 | 769.63 | 765.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 754.85 | 769.63 | 765.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 754.85 | 769.63 | 765.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 754.85 | 769.63 | 765.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 750.80 | 765.86 | 763.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:45:00 | 750.30 | 765.86 | 763.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 747.35 | 762.16 | 762.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 740.65 | 757.86 | 760.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 725.55 | 724.17 | 733.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 12:45:00 | 726.85 | 724.17 | 733.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 715.00 | 721.25 | 729.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 736.65 | 725.32 | 730.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 741.60 | 728.58 | 731.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:30:00 | 742.45 | 728.58 | 731.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 751.30 | 735.80 | 734.45 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 728.00 | 736.27 | 737.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 722.85 | 732.15 | 734.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 718.40 | 715.98 | 720.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 718.40 | 715.98 | 720.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 720.50 | 716.89 | 720.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 720.95 | 716.89 | 720.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 722.35 | 717.98 | 720.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 723.35 | 717.98 | 720.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 721.65 | 718.71 | 720.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 726.40 | 718.71 | 720.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 723.00 | 719.57 | 721.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 736.00 | 719.57 | 721.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 741.30 | 723.92 | 723.00 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 714.80 | 726.16 | 726.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 708.65 | 718.90 | 722.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 723.25 | 709.15 | 714.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 723.25 | 709.15 | 714.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 723.25 | 709.15 | 714.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 723.25 | 709.15 | 714.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 712.10 | 709.74 | 714.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 703.90 | 715.66 | 715.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 725.30 | 714.38 | 714.49 | SL hit (close>static) qty=1.00 sl=723.90 alert=retest2 |

### Cycle 151 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 728.95 | 717.29 | 715.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 734.20 | 726.01 | 721.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 757.55 | 759.01 | 748.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 757.55 | 759.01 | 748.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 753.75 | 760.23 | 752.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:00:00 | 753.75 | 760.23 | 752.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 754.10 | 759.01 | 752.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:45:00 | 752.55 | 759.01 | 752.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 757.50 | 766.53 | 761.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 758.80 | 766.53 | 761.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 762.90 | 765.33 | 761.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 769.55 | 762.97 | 761.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 789.80 | 798.46 | 799.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 789.80 | 798.46 | 799.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 784.90 | 795.75 | 797.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 800.00 | 793.48 | 796.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 800.00 | 793.48 | 796.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 800.00 | 793.48 | 796.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 800.00 | 793.48 | 796.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 788.15 | 792.41 | 795.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:30:00 | 799.60 | 792.41 | 795.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 800.00 | 793.93 | 795.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 800.00 | 793.93 | 795.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 800.00 | 795.14 | 796.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 15:15:00 | 794.50 | 795.14 | 796.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 811.55 | 798.32 | 797.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 811.55 | 798.32 | 797.41 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 15:15:00 | 793.55 | 797.84 | 798.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 785.50 | 795.37 | 796.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 15:15:00 | 789.60 | 789.15 | 792.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:15:00 | 786.95 | 789.15 | 792.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 787.25 | 788.77 | 792.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:45:00 | 781.15 | 786.02 | 789.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 12:15:00 | 769.50 | 762.58 | 761.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 769.50 | 762.58 | 761.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 790.50 | 768.16 | 764.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 784.75 | 787.02 | 779.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 784.75 | 787.02 | 779.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 784.75 | 787.02 | 779.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 783.30 | 787.02 | 779.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-13 09:15:00 | 850.05 | 2024-06-13 13:15:00 | 833.20 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-06-13 10:30:00 | 847.95 | 2024-06-13 13:15:00 | 833.20 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-06-13 12:15:00 | 845.20 | 2024-06-13 13:15:00 | 833.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest1 | 2024-06-25 09:15:00 | 844.25 | 2024-06-25 12:15:00 | 833.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-06-27 11:15:00 | 807.55 | 2024-07-01 13:15:00 | 841.95 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2024-06-27 13:30:00 | 810.75 | 2024-07-01 13:15:00 | 841.95 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2024-06-28 10:00:00 | 813.05 | 2024-07-01 13:15:00 | 841.95 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2024-06-28 11:45:00 | 812.95 | 2024-07-01 13:15:00 | 841.95 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2024-07-03 09:15:00 | 860.10 | 2024-07-08 12:15:00 | 846.10 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-07-04 12:00:00 | 851.40 | 2024-07-08 12:15:00 | 846.10 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-07-04 12:45:00 | 850.50 | 2024-07-08 13:15:00 | 846.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-07-05 10:15:00 | 853.25 | 2024-07-08 13:15:00 | 846.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-07-08 09:15:00 | 861.00 | 2024-07-08 13:15:00 | 846.90 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-07-08 10:00:00 | 858.70 | 2024-07-08 13:15:00 | 846.90 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-07-11 11:00:00 | 840.70 | 2024-07-15 11:15:00 | 798.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 11:00:00 | 840.70 | 2024-07-19 09:15:00 | 756.63 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-07-24 09:15:00 | 809.00 | 2024-08-01 12:15:00 | 841.45 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest2 | 2024-07-24 09:45:00 | 809.95 | 2024-08-01 12:15:00 | 841.45 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2024-07-24 10:45:00 | 809.00 | 2024-08-01 12:15:00 | 841.45 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest2 | 2024-07-24 13:15:00 | 812.10 | 2024-08-01 12:15:00 | 841.45 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2024-07-26 14:00:00 | 834.75 | 2024-08-01 12:15:00 | 841.45 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2024-07-30 09:15:00 | 855.50 | 2024-08-01 12:15:00 | 841.45 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest1 | 2024-08-06 14:00:00 | 790.55 | 2024-08-07 14:15:00 | 802.65 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-08 09:15:00 | 799.55 | 2024-08-21 15:15:00 | 776.00 | STOP_HIT | 1.00 | 2.95% |
| SELL | retest2 | 2024-08-08 09:45:00 | 796.05 | 2024-08-21 15:15:00 | 776.00 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2024-08-08 10:45:00 | 799.10 | 2024-08-21 15:15:00 | 776.00 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2024-09-09 09:15:00 | 899.30 | 2024-09-09 13:15:00 | 881.15 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-09-10 10:15:00 | 896.30 | 2024-09-11 14:15:00 | 883.65 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-09-16 09:15:00 | 902.60 | 2024-09-16 13:15:00 | 881.40 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-09-24 11:15:00 | 902.95 | 2024-09-26 09:15:00 | 868.00 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2024-10-04 13:30:00 | 870.00 | 2024-10-07 15:15:00 | 826.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 14:00:00 | 865.95 | 2024-10-08 09:15:00 | 822.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 13:30:00 | 870.00 | 2024-10-08 10:15:00 | 865.45 | STOP_HIT | 0.50 | 0.52% |
| SELL | retest2 | 2024-10-04 14:00:00 | 865.95 | 2024-10-08 10:15:00 | 865.45 | STOP_HIT | 0.50 | 0.06% |
| BUY | retest2 | 2024-10-11 12:15:00 | 881.60 | 2024-10-16 13:15:00 | 886.90 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2024-10-21 09:15:00 | 868.20 | 2024-10-23 11:15:00 | 824.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 869.90 | 2024-10-23 11:15:00 | 826.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 868.20 | 2024-10-23 15:15:00 | 841.65 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2024-10-21 14:00:00 | 869.90 | 2024-10-23 15:15:00 | 841.65 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2024-10-29 10:45:00 | 821.80 | 2024-10-30 14:15:00 | 837.65 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-11-11 09:30:00 | 920.45 | 2024-11-11 15:15:00 | 879.00 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2024-11-18 09:30:00 | 831.95 | 2024-11-19 11:15:00 | 851.80 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-11-18 12:45:00 | 838.25 | 2024-11-19 11:15:00 | 851.80 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-12-02 09:30:00 | 894.05 | 2024-12-02 10:15:00 | 881.10 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-05 09:15:00 | 899.10 | 2024-12-05 09:15:00 | 889.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-12-05 14:45:00 | 894.25 | 2024-12-05 15:15:00 | 890.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-12-06 09:15:00 | 903.50 | 2024-12-12 12:15:00 | 989.01 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2024-12-09 09:45:00 | 899.10 | 2024-12-12 14:15:00 | 993.85 | TARGET_HIT | 1.00 | 10.54% |
| BUY | retest2 | 2024-12-12 09:15:00 | 951.25 | 2024-12-13 09:15:00 | 1046.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-24 12:15:00 | 961.05 | 2024-12-24 14:15:00 | 979.80 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-12-27 13:15:00 | 987.75 | 2024-12-31 10:15:00 | 957.65 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-12-31 09:45:00 | 983.45 | 2024-12-31 10:15:00 | 957.65 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-01-03 14:45:00 | 996.25 | 2025-01-06 09:15:00 | 985.40 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-01-03 15:15:00 | 1000.00 | 2025-01-06 09:15:00 | 985.40 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-01-23 14:30:00 | 783.25 | 2025-01-27 09:15:00 | 744.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:30:00 | 783.25 | 2025-01-28 12:15:00 | 738.50 | STOP_HIT | 0.50 | 5.71% |
| BUY | retest2 | 2025-02-01 14:15:00 | 764.40 | 2025-02-03 11:15:00 | 840.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-11 12:15:00 | 704.80 | 2025-02-14 09:15:00 | 669.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 12:15:00 | 704.80 | 2025-02-14 11:15:00 | 684.60 | STOP_HIT | 0.50 | 2.87% |
| BUY | retest2 | 2025-02-21 11:30:00 | 722.70 | 2025-02-28 11:15:00 | 702.95 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-02-21 14:15:00 | 717.75 | 2025-02-28 11:15:00 | 702.95 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-02-25 09:45:00 | 719.65 | 2025-02-28 11:15:00 | 702.95 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest1 | 2025-03-07 11:15:00 | 780.65 | 2025-03-10 14:15:00 | 819.68 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-07 11:15:00 | 780.65 | 2025-03-10 14:15:00 | 791.10 | STOP_HIT | 0.50 | 1.34% |
| BUY | retest2 | 2025-03-11 12:00:00 | 786.40 | 2025-03-11 15:15:00 | 772.15 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-03-13 09:15:00 | 765.50 | 2025-03-19 09:15:00 | 799.15 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-03-18 10:30:00 | 775.95 | 2025-03-19 09:15:00 | 799.15 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2025-03-26 09:15:00 | 867.90 | 2025-03-26 14:15:00 | 844.60 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-03-27 09:15:00 | 864.40 | 2025-03-27 14:15:00 | 838.55 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-04-01 12:15:00 | 825.45 | 2025-04-01 14:15:00 | 840.55 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-09 09:15:00 | 806.50 | 2025-04-11 15:15:00 | 825.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-04-09 10:00:00 | 810.55 | 2025-04-11 15:15:00 | 825.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-04-16 11:45:00 | 827.00 | 2025-04-17 11:15:00 | 821.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-04-16 13:45:00 | 826.10 | 2025-04-17 11:15:00 | 821.90 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-05-16 14:00:00 | 903.55 | 2025-05-20 14:15:00 | 894.70 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-05-16 15:00:00 | 903.85 | 2025-05-20 14:15:00 | 894.70 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-19 10:00:00 | 908.05 | 2025-05-20 14:15:00 | 894.70 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-05-23 13:15:00 | 887.25 | 2025-05-26 09:15:00 | 898.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-05-23 15:00:00 | 886.85 | 2025-05-26 09:15:00 | 898.95 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-05-28 12:15:00 | 921.80 | 2025-06-03 11:15:00 | 916.45 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-05-28 13:30:00 | 920.00 | 2025-06-03 11:15:00 | 916.45 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-05-30 10:00:00 | 921.15 | 2025-06-03 11:15:00 | 916.45 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-02 09:15:00 | 928.00 | 2025-06-03 11:15:00 | 916.45 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-06-04 11:30:00 | 915.80 | 2025-06-06 09:15:00 | 922.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-04 12:00:00 | 915.00 | 2025-06-06 09:15:00 | 922.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-06-04 15:00:00 | 915.40 | 2025-06-06 09:15:00 | 922.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-05 09:45:00 | 915.40 | 2025-06-06 09:15:00 | 922.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-06-09 09:15:00 | 925.55 | 2025-06-10 12:15:00 | 912.05 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-06-12 12:45:00 | 898.05 | 2025-06-17 13:15:00 | 895.75 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-06-25 09:45:00 | 908.50 | 2025-07-01 14:15:00 | 906.90 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-07-07 12:15:00 | 885.00 | 2025-07-10 10:15:00 | 881.30 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-08-07 13:15:00 | 873.85 | 2025-08-12 11:15:00 | 880.30 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-08-08 10:00:00 | 874.80 | 2025-08-12 11:15:00 | 880.30 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-08-08 14:00:00 | 873.20 | 2025-08-12 11:15:00 | 880.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-11 09:45:00 | 871.00 | 2025-08-12 11:15:00 | 880.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-08-26 12:15:00 | 1023.75 | 2025-08-26 15:15:00 | 1003.45 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest1 | 2025-09-10 09:15:00 | 1036.90 | 2025-09-11 10:15:00 | 1031.50 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1020.70 | 2025-09-26 09:15:00 | 969.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:45:00 | 1021.00 | 2025-09-26 09:15:00 | 969.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:00:00 | 1022.50 | 2025-09-26 09:15:00 | 971.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1020.70 | 2025-09-30 09:15:00 | 967.80 | STOP_HIT | 0.50 | 5.18% |
| SELL | retest2 | 2025-09-24 10:45:00 | 1021.00 | 2025-09-30 09:15:00 | 967.80 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2025-09-24 14:00:00 | 1022.50 | 2025-09-30 09:15:00 | 967.80 | STOP_HIT | 0.50 | 5.35% |
| SELL | retest1 | 2025-10-09 11:45:00 | 933.45 | 2025-10-10 09:15:00 | 950.30 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest1 | 2025-10-09 13:30:00 | 939.35 | 2025-10-10 09:15:00 | 950.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-10-10 12:15:00 | 957.15 | 2025-10-10 15:15:00 | 959.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-10-23 14:30:00 | 952.00 | 2025-10-29 14:15:00 | 951.20 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-10-23 15:15:00 | 950.80 | 2025-10-29 14:15:00 | 951.20 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-10-24 10:00:00 | 952.10 | 2025-10-29 14:15:00 | 951.20 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-10-24 11:15:00 | 948.20 | 2025-10-29 14:15:00 | 951.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-10-24 13:00:00 | 942.70 | 2025-10-29 14:15:00 | 951.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-10-27 09:30:00 | 942.50 | 2025-10-29 14:15:00 | 951.20 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-10-27 15:15:00 | 944.50 | 2025-10-29 14:15:00 | 951.20 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-13 14:45:00 | 890.10 | 2025-11-19 12:15:00 | 885.70 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-12-01 11:00:00 | 897.00 | 2025-12-02 09:15:00 | 884.80 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-10 14:30:00 | 887.90 | 2025-12-12 09:15:00 | 892.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-11 15:00:00 | 883.90 | 2025-12-12 09:15:00 | 892.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-12-17 12:15:00 | 871.70 | 2025-12-22 09:15:00 | 872.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-12-30 09:15:00 | 865.25 | 2025-12-31 11:15:00 | 874.95 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-31 11:00:00 | 868.50 | 2025-12-31 11:15:00 | 874.95 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-09 09:15:00 | 866.15 | 2026-01-09 09:15:00 | 878.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-12 09:15:00 | 856.30 | 2026-01-13 14:15:00 | 873.10 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-01-22 11:30:00 | 849.40 | 2026-01-27 09:15:00 | 806.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 849.40 | 2026-01-27 15:15:00 | 821.90 | STOP_HIT | 0.50 | 3.24% |
| BUY | retest2 | 2026-02-11 14:00:00 | 877.40 | 2026-02-13 09:15:00 | 857.20 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-02-12 10:00:00 | 880.80 | 2026-02-13 09:15:00 | 857.20 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-02-17 09:15:00 | 852.50 | 2026-02-17 10:15:00 | 868.65 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-02-17 10:00:00 | 855.80 | 2026-02-17 10:15:00 | 868.65 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-03-06 09:30:00 | 762.85 | 2026-03-11 11:15:00 | 769.55 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-03-06 15:00:00 | 762.90 | 2026-03-11 11:15:00 | 769.55 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-03-09 09:15:00 | 739.55 | 2026-03-11 11:15:00 | 769.55 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2026-04-02 09:15:00 | 703.90 | 2026-04-02 13:15:00 | 725.30 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-04-13 10:15:00 | 758.80 | 2026-04-23 14:15:00 | 789.80 | STOP_HIT | 1.00 | 4.09% |
| BUY | retest2 | 2026-04-13 10:45:00 | 762.90 | 2026-04-23 14:15:00 | 789.80 | STOP_HIT | 1.00 | 3.53% |
| BUY | retest2 | 2026-04-15 09:15:00 | 769.55 | 2026-04-23 14:15:00 | 789.80 | STOP_HIT | 1.00 | 2.63% |
| SELL | retest2 | 2026-04-24 15:15:00 | 794.50 | 2026-04-27 09:15:00 | 811.55 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-29 12:45:00 | 781.15 | 2026-05-06 12:15:00 | 769.50 | STOP_HIT | 1.00 | 1.49% |
