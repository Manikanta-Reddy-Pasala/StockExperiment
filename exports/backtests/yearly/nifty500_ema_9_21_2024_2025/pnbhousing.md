# PNB Housing Finance Ltd. (PNBHOUSING)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1088.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 127 |
| ALERT1 | 92 |
| ALERT2 | 91 |
| ALERT2_SKIP | 37 |
| ALERT3 | 281 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 133 |
| PARTIAL | 10 |
| TARGET_HIT | 14 |
| STOP_HIT | 121 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 41 / 100
- **Target hits / Stop hits / Partials:** 14 / 119 / 8
- **Avg / median % per leg:** 0.43% / -0.92%
- **Sum % (uncompounded):** 60.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 18 | 23.7% | 9 | 67 | 0 | 0.23% | 17.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.16% | -3.5% |
| BUY @ 3rd Alert (retest2) | 73 | 18 | 24.7% | 9 | 64 | 0 | 0.28% | 20.6% |
| SELL (all) | 65 | 23 | 35.4% | 5 | 52 | 8 | 0.67% | 43.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 65 | 23 | 35.4% | 5 | 52 | 8 | 0.67% | 43.2% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.16% | -3.5% |
| retest2 (combined) | 138 | 41 | 29.7% | 14 | 116 | 8 | 0.46% | 63.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 744.75 | 739.16 | 738.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 746.80 | 739.98 | 739.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 14:15:00 | 745.50 | 745.68 | 742.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 15:00:00 | 745.50 | 745.68 | 742.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 742.00 | 744.95 | 742.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 747.50 | 744.95 | 742.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 13:15:00 | 736.80 | 743.73 | 743.11 | SL hit (close<static) qty=1.00 sl=739.95 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 14:15:00 | 736.05 | 742.20 | 742.47 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 748.00 | 743.46 | 742.93 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 738.80 | 742.78 | 742.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 12:15:00 | 737.30 | 741.13 | 742.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 14:15:00 | 741.85 | 741.26 | 741.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 14:15:00 | 741.85 | 741.26 | 741.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 741.85 | 741.26 | 741.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:45:00 | 739.85 | 741.26 | 741.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 744.40 | 741.98 | 742.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 744.40 | 741.98 | 742.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 741.30 | 741.85 | 742.07 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 13:15:00 | 746.60 | 743.03 | 742.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 767.50 | 748.51 | 745.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 15:15:00 | 793.00 | 794.78 | 782.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:45:00 | 799.00 | 795.83 | 785.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 12:00:00 | 800.90 | 796.85 | 786.45 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 15:00:00 | 800.00 | 798.97 | 790.15 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 790.70 | 797.80 | 791.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 790.70 | 797.80 | 791.18 | SL hit (close<ema400) qty=1.00 sl=791.18 alert=retest1 |

### Cycle 6 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 739.70 | 781.94 | 786.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 721.35 | 740.07 | 751.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 736.05 | 729.55 | 739.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 736.05 | 729.55 | 739.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 736.05 | 729.55 | 739.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:00:00 | 726.60 | 728.96 | 738.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:30:00 | 730.20 | 730.57 | 736.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:45:00 | 729.60 | 729.98 | 735.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 703.55 | 730.83 | 735.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 711.20 | 726.91 | 733.52 | EMA400 retest candle locked (from downside) |
| Target hit | 2024-06-04 10:15:00 | 653.94 | 712.30 | 726.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 731.00 | 708.25 | 707.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 739.10 | 714.42 | 710.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 811.05 | 811.95 | 797.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 10:45:00 | 810.60 | 811.95 | 797.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 873.80 | 866.96 | 850.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 857.90 | 866.96 | 850.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 842.45 | 865.40 | 854.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 842.45 | 865.40 | 854.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 847.75 | 861.87 | 853.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:30:00 | 843.95 | 861.87 | 853.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 805.60 | 844.17 | 847.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 14:15:00 | 783.80 | 794.47 | 809.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 11:15:00 | 774.10 | 773.39 | 785.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 11:30:00 | 774.45 | 773.39 | 785.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 782.45 | 775.18 | 783.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:45:00 | 778.95 | 775.18 | 783.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 781.55 | 776.45 | 782.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 795.85 | 776.45 | 782.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 795.40 | 780.24 | 784.02 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 796.15 | 788.21 | 787.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 801.95 | 792.92 | 789.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 789.95 | 795.67 | 792.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 789.95 | 795.67 | 792.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 789.95 | 795.67 | 792.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 789.95 | 795.67 | 792.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 784.35 | 793.40 | 791.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 784.35 | 793.40 | 791.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 789.10 | 792.00 | 791.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 784.45 | 792.00 | 791.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 10:15:00 | 783.30 | 790.26 | 790.68 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 795.35 | 789.44 | 789.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 797.00 | 790.96 | 789.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 793.75 | 795.63 | 793.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 10:15:00 | 793.75 | 795.63 | 793.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 793.75 | 795.63 | 793.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:00:00 | 793.75 | 795.63 | 793.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 791.20 | 794.75 | 792.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 791.20 | 794.75 | 792.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 787.50 | 793.30 | 792.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 787.50 | 793.30 | 792.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 795.85 | 793.63 | 792.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 810.30 | 793.71 | 792.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 804.00 | 794.81 | 793.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:45:00 | 802.20 | 797.78 | 795.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 14:30:00 | 803.15 | 799.27 | 796.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 798.65 | 799.26 | 797.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 809.30 | 800.82 | 798.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 14:15:00 | 793.10 | 798.34 | 798.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 14:15:00 | 793.10 | 798.34 | 798.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 783.65 | 794.39 | 796.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 13:15:00 | 796.30 | 792.34 | 794.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 13:15:00 | 796.30 | 792.34 | 794.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 796.30 | 792.34 | 794.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 796.30 | 792.34 | 794.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 791.70 | 792.22 | 794.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 13:00:00 | 790.20 | 792.65 | 794.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:45:00 | 789.00 | 791.08 | 792.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 13:15:00 | 790.70 | 789.51 | 791.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 14:15:00 | 809.00 | 793.54 | 793.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 809.00 | 793.54 | 793.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 812.95 | 802.92 | 798.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 802.45 | 806.80 | 803.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 802.45 | 806.80 | 803.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 802.45 | 806.80 | 803.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 799.00 | 806.80 | 803.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 805.20 | 806.48 | 803.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:45:00 | 804.80 | 806.48 | 803.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 803.35 | 805.85 | 803.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:30:00 | 805.30 | 805.85 | 803.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 805.00 | 805.68 | 803.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:45:00 | 804.90 | 805.68 | 803.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 800.90 | 805.22 | 803.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 800.90 | 805.22 | 803.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 802.60 | 804.70 | 803.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:45:00 | 806.20 | 804.34 | 803.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 813.50 | 806.17 | 804.44 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 796.10 | 803.72 | 804.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 786.30 | 798.73 | 801.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 787.90 | 784.51 | 789.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 787.90 | 784.51 | 789.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 787.90 | 784.51 | 789.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 787.90 | 784.51 | 789.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 784.00 | 784.83 | 789.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 794.60 | 784.83 | 789.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 789.50 | 784.58 | 788.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 787.80 | 784.58 | 788.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 792.05 | 786.07 | 788.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 787.20 | 786.07 | 788.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 794.95 | 788.70 | 789.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 794.95 | 788.70 | 789.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 780.00 | 786.96 | 788.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 749.05 | 786.96 | 788.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:00:00 | 778.40 | 770.96 | 775.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 15:00:00 | 773.90 | 771.55 | 775.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 799.45 | 778.65 | 776.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 799.45 | 778.65 | 776.40 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 781.90 | 787.19 | 787.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 14:15:00 | 779.00 | 784.55 | 786.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 09:15:00 | 790.50 | 784.85 | 785.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 790.50 | 784.85 | 785.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 790.50 | 784.85 | 785.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:00:00 | 790.50 | 784.85 | 785.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 790.50 | 785.98 | 786.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 790.50 | 785.98 | 786.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 11:15:00 | 790.45 | 786.87 | 786.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 12:15:00 | 806.95 | 790.89 | 788.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 13:15:00 | 810.05 | 810.24 | 801.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 14:00:00 | 810.05 | 810.24 | 801.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 809.85 | 810.16 | 802.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:30:00 | 817.60 | 806.44 | 803.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 15:15:00 | 811.50 | 806.95 | 803.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 788.20 | 803.93 | 803.06 | SL hit (close<static) qty=1.00 sl=800.90 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 774.05 | 797.95 | 800.42 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 13:15:00 | 799.60 | 794.18 | 794.17 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 793.50 | 794.05 | 794.11 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 808.45 | 796.15 | 794.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 11:15:00 | 829.30 | 814.27 | 808.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 833.00 | 833.45 | 825.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 10:00:00 | 833.00 | 833.45 | 825.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 813.60 | 828.45 | 825.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 813.60 | 828.45 | 825.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 813.90 | 825.54 | 824.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 813.90 | 825.54 | 824.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 803.15 | 821.06 | 822.30 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 828.80 | 822.16 | 821.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 831.45 | 825.14 | 823.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 14:15:00 | 825.95 | 828.99 | 826.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 825.95 | 828.99 | 826.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 825.95 | 828.99 | 826.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 825.95 | 828.99 | 826.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 824.60 | 828.11 | 825.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 829.00 | 828.11 | 825.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 10:30:00 | 826.10 | 826.74 | 825.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 11:45:00 | 826.15 | 826.31 | 825.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 12:15:00 | 821.40 | 825.33 | 825.19 | SL hit (close<static) qty=1.00 sl=821.70 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 821.15 | 824.49 | 824.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 816.40 | 822.44 | 823.77 | Break + close below crossover candle low |

### Cycle 25 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 871.85 | 826.48 | 823.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 914.50 | 884.74 | 875.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 13:15:00 | 970.55 | 970.56 | 954.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:45:00 | 970.10 | 970.56 | 954.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 957.10 | 968.45 | 957.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 957.30 | 968.45 | 957.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 991.00 | 972.96 | 960.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:30:00 | 958.45 | 972.96 | 960.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1016.80 | 996.40 | 979.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:15:00 | 1046.55 | 1008.09 | 988.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 12:45:00 | 1039.55 | 1043.72 | 1023.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 13:15:00 | 1034.70 | 1043.72 | 1023.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 10:45:00 | 1043.35 | 1036.60 | 1027.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 1028.95 | 1035.07 | 1027.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:45:00 | 1025.85 | 1035.07 | 1027.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 1023.25 | 1032.71 | 1026.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 1023.25 | 1032.71 | 1026.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 1025.30 | 1031.22 | 1026.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:30:00 | 1022.35 | 1031.22 | 1026.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1021.95 | 1029.37 | 1026.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 1018.15 | 1029.37 | 1026.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1041.35 | 1030.76 | 1027.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 1040.70 | 1030.76 | 1027.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 1042.05 | 1036.52 | 1031.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 12:30:00 | 1044.90 | 1036.52 | 1031.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 1055.75 | 1059.11 | 1048.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:30:00 | 1054.15 | 1059.11 | 1048.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 1046.30 | 1055.67 | 1048.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 1046.30 | 1055.67 | 1048.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 1049.60 | 1054.45 | 1048.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 1058.00 | 1054.45 | 1048.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1078.65 | 1059.29 | 1051.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:15:00 | 1105.75 | 1059.29 | 1051.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-13 09:15:00 | 1151.21 | 1114.22 | 1095.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 1067.10 | 1108.68 | 1112.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 1053.00 | 1084.61 | 1099.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 1075.25 | 1059.45 | 1080.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 13:30:00 | 1067.95 | 1059.45 | 1080.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1084.00 | 1061.80 | 1075.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:45:00 | 1049.85 | 1062.84 | 1070.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 14:15:00 | 997.36 | 1008.84 | 1016.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 14:15:00 | 1000.25 | 991.07 | 1001.29 | SL hit (close>ema200) qty=0.50 sl=991.07 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 966.35 | 950.79 | 949.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 12:15:00 | 979.70 | 971.42 | 964.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 971.75 | 983.65 | 975.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 12:15:00 | 971.75 | 983.65 | 975.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 971.75 | 983.65 | 975.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:45:00 | 968.20 | 983.65 | 975.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 974.50 | 981.82 | 975.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:15:00 | 975.70 | 981.82 | 975.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 15:00:00 | 980.75 | 981.61 | 976.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 969.30 | 978.74 | 975.81 | SL hit (close<static) qty=1.00 sl=971.05 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 957.65 | 972.51 | 973.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 939.65 | 959.80 | 966.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 930.60 | 926.40 | 938.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 15:00:00 | 930.60 | 926.40 | 938.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 934.55 | 928.03 | 938.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 935.50 | 928.03 | 938.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 937.80 | 929.99 | 938.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 918.90 | 926.77 | 934.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 872.95 | 897.07 | 913.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 914.45 | 897.65 | 910.80 | SL hit (close>ema200) qty=0.50 sl=897.65 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 953.25 | 919.60 | 918.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 09:15:00 | 961.35 | 937.07 | 928.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 14:15:00 | 932.65 | 947.96 | 938.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 14:15:00 | 932.65 | 947.96 | 938.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 932.65 | 947.96 | 938.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 932.65 | 947.96 | 938.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 932.00 | 944.77 | 937.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 988.25 | 944.77 | 937.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 10:00:00 | 943.45 | 951.84 | 947.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 928.25 | 943.97 | 944.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 11:15:00 | 928.25 | 943.97 | 944.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 12:15:00 | 920.95 | 939.37 | 942.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 13:15:00 | 914.60 | 914.43 | 924.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 13:30:00 | 914.00 | 914.43 | 924.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 919.80 | 916.54 | 923.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 930.90 | 916.54 | 923.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 929.45 | 919.13 | 924.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 933.55 | 919.13 | 924.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 930.10 | 921.32 | 924.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:45:00 | 932.70 | 921.32 | 924.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 924.60 | 924.26 | 925.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 933.95 | 924.26 | 925.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 935.90 | 926.59 | 926.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 940.85 | 931.03 | 928.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 12:15:00 | 964.60 | 965.09 | 952.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 13:00:00 | 964.60 | 965.09 | 952.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 955.60 | 963.19 | 952.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:00:00 | 955.60 | 963.19 | 952.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 950.25 | 960.60 | 952.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 950.25 | 960.60 | 952.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 948.00 | 958.08 | 952.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 950.70 | 958.08 | 952.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 956.95 | 958.12 | 953.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 956.95 | 958.12 | 953.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 954.20 | 957.33 | 953.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:30:00 | 951.00 | 957.33 | 953.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 955.10 | 956.89 | 953.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:30:00 | 951.60 | 956.89 | 953.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 965.05 | 958.52 | 954.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:15:00 | 968.10 | 958.52 | 954.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 962.50 | 976.46 | 976.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 962.50 | 976.46 | 976.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 955.95 | 969.98 | 973.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 1002.80 | 966.83 | 969.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 1002.80 | 966.83 | 969.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1002.80 | 966.83 | 969.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:00:00 | 1002.80 | 966.83 | 969.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 994.80 | 972.42 | 971.87 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 963.90 | 971.30 | 971.99 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 979.55 | 972.95 | 972.67 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 921.45 | 965.09 | 969.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 15:15:00 | 897.35 | 923.85 | 938.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 887.70 | 885.15 | 905.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 09:30:00 | 889.50 | 885.15 | 905.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 870.95 | 857.68 | 865.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 873.15 | 857.68 | 865.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 872.75 | 860.69 | 866.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:30:00 | 872.50 | 860.69 | 866.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 865.15 | 863.14 | 865.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 865.15 | 863.14 | 865.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 865.10 | 863.53 | 865.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 865.00 | 863.53 | 865.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 854.90 | 861.81 | 864.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 10:15:00 | 851.05 | 861.81 | 864.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 12:15:00 | 869.25 | 863.86 | 863.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 869.25 | 863.86 | 863.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 14:15:00 | 874.70 | 867.44 | 865.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 13:15:00 | 874.60 | 874.85 | 870.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 14:00:00 | 874.60 | 874.85 | 870.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 873.95 | 877.65 | 874.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 874.10 | 877.65 | 874.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 875.50 | 877.22 | 874.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 873.20 | 877.22 | 874.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 882.30 | 878.24 | 874.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:30:00 | 885.60 | 880.79 | 876.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 10:15:00 | 873.10 | 879.57 | 877.02 | SL hit (close<static) qty=1.00 sl=873.65 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 940.70 | 950.97 | 951.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 933.95 | 944.44 | 948.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 930.55 | 928.98 | 934.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 930.55 | 928.98 | 934.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 931.95 | 931.10 | 934.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 934.00 | 931.10 | 934.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 872.25 | 867.80 | 875.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 873.35 | 867.80 | 875.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 864.65 | 867.17 | 874.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 859.45 | 867.17 | 874.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 10:45:00 | 859.15 | 865.49 | 873.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 858.15 | 852.23 | 851.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 858.15 | 852.23 | 851.86 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 13:15:00 | 850.30 | 851.50 | 851.60 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 852.70 | 851.74 | 851.70 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 845.35 | 850.46 | 851.12 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 858.55 | 851.70 | 851.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 871.70 | 855.45 | 853.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 851.10 | 855.79 | 853.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 851.10 | 855.79 | 853.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 851.10 | 855.79 | 853.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:30:00 | 852.50 | 855.79 | 853.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 849.45 | 854.52 | 853.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 849.45 | 854.52 | 853.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 850.80 | 853.78 | 853.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 851.70 | 853.78 | 853.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 850.40 | 853.10 | 852.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 849.95 | 853.10 | 852.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 919.15 | 925.23 | 916.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 923.65 | 925.23 | 916.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 914.30 | 923.05 | 915.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 914.30 | 923.05 | 915.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 910.90 | 920.62 | 915.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 910.65 | 920.62 | 915.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 911.40 | 918.77 | 915.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:30:00 | 913.90 | 918.77 | 915.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 900.05 | 915.03 | 913.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 897.15 | 915.03 | 913.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 903.00 | 912.62 | 912.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 886.00 | 907.30 | 910.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 878.75 | 876.47 | 885.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 09:30:00 | 878.25 | 876.47 | 885.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 885.85 | 878.13 | 884.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:30:00 | 886.20 | 878.13 | 884.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 888.90 | 880.29 | 885.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:00:00 | 888.90 | 880.29 | 885.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 890.60 | 882.35 | 885.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 890.60 | 882.35 | 885.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 890.10 | 883.90 | 886.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 890.10 | 883.90 | 886.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 859.75 | 879.58 | 883.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 14:00:00 | 845.00 | 863.81 | 874.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 15:15:00 | 851.00 | 861.91 | 872.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 802.75 | 842.81 | 859.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 808.45 | 842.81 | 859.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 823.60 | 822.85 | 841.56 | SL hit (close>ema200) qty=0.50 sl=822.85 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 850.00 | 843.33 | 843.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 859.45 | 846.56 | 844.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 896.20 | 897.72 | 884.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 09:45:00 | 898.90 | 897.72 | 884.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 899.00 | 904.60 | 898.45 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 889.40 | 895.71 | 896.00 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 917.25 | 898.47 | 897.03 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 11:15:00 | 887.50 | 895.75 | 896.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 15:15:00 | 878.00 | 890.61 | 893.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 817.15 | 814.82 | 835.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 817.15 | 814.82 | 835.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 854.95 | 822.84 | 837.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 854.95 | 822.84 | 837.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 845.90 | 827.45 | 838.35 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 859.15 | 844.21 | 843.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 12:15:00 | 861.00 | 847.57 | 844.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 861.15 | 863.02 | 855.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 861.15 | 863.02 | 855.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 854.15 | 860.88 | 855.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 854.20 | 860.88 | 855.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 866.60 | 862.02 | 856.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 857.10 | 862.02 | 856.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 864.00 | 862.42 | 857.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 867.25 | 862.42 | 857.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 868.15 | 863.56 | 858.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 879.80 | 863.56 | 858.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:45:00 | 878.00 | 869.55 | 862.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 13:15:00 | 877.25 | 869.55 | 862.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 885.10 | 871.39 | 864.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 899.40 | 888.22 | 877.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-04 11:15:00 | 862.00 | 876.08 | 877.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 11:15:00 | 862.00 | 876.08 | 877.66 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 887.85 | 877.66 | 877.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 906.05 | 886.19 | 881.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 11:15:00 | 902.15 | 907.59 | 899.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 12:00:00 | 902.15 | 907.59 | 899.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 887.80 | 902.98 | 899.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 887.80 | 902.98 | 899.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 890.65 | 900.51 | 898.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:15:00 | 888.90 | 900.51 | 898.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 878.90 | 894.33 | 895.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 858.50 | 887.17 | 892.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 821.00 | 819.95 | 836.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 819.95 | 819.95 | 836.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 832.45 | 822.45 | 835.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 832.45 | 822.45 | 835.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 835.70 | 825.10 | 835.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 835.90 | 825.10 | 835.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 836.50 | 827.38 | 835.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 839.50 | 827.38 | 835.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 820.10 | 825.92 | 834.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 818.00 | 825.92 | 834.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 777.10 | 792.66 | 806.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-18 10:15:00 | 783.55 | 783.10 | 792.83 | SL hit (close>ema200) qty=0.50 sl=783.10 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 810.90 | 795.93 | 795.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 820.60 | 803.70 | 799.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 813.80 | 819.88 | 813.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 813.80 | 819.88 | 813.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 813.80 | 819.88 | 813.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 813.80 | 819.88 | 813.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 813.80 | 818.66 | 813.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 813.80 | 818.66 | 813.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 807.80 | 816.49 | 813.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 807.80 | 816.49 | 813.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 806.05 | 814.40 | 812.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:00:00 | 806.05 | 814.40 | 812.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 806.00 | 811.15 | 811.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 794.30 | 807.78 | 809.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 11:15:00 | 796.50 | 794.56 | 799.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 11:45:00 | 795.45 | 794.56 | 799.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 799.70 | 795.59 | 799.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:00:00 | 799.70 | 795.59 | 799.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 795.40 | 795.55 | 799.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 795.40 | 795.55 | 799.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 777.00 | 790.50 | 795.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:30:00 | 763.45 | 777.14 | 785.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 11:15:00 | 791.90 | 774.80 | 772.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 791.90 | 774.80 | 772.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 800.45 | 789.53 | 782.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 805.35 | 809.32 | 799.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 812.45 | 809.32 | 799.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 818.50 | 816.41 | 809.31 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 787.05 | 803.62 | 805.35 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 804.15 | 800.32 | 800.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 819.15 | 804.09 | 801.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 13:15:00 | 808.20 | 808.84 | 805.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 14:15:00 | 804.60 | 808.84 | 805.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 802.70 | 807.61 | 804.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:45:00 | 802.80 | 807.61 | 804.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 801.00 | 806.29 | 804.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 818.25 | 806.29 | 804.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 801.50 | 803.94 | 804.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 13:15:00 | 801.50 | 803.94 | 804.05 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 810.75 | 804.83 | 804.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 819.70 | 807.86 | 805.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 824.10 | 828.57 | 822.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 14:15:00 | 824.10 | 828.57 | 822.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 824.10 | 828.57 | 822.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 15:00:00 | 824.10 | 828.57 | 822.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 823.00 | 827.46 | 822.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 839.95 | 827.46 | 822.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 12:15:00 | 857.35 | 866.59 | 867.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 857.35 | 866.59 | 867.12 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 873.45 | 867.72 | 867.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 15:15:00 | 877.95 | 869.77 | 868.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 890.15 | 931.74 | 922.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 890.15 | 931.74 | 922.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 890.15 | 931.74 | 922.47 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 879.05 | 914.62 | 915.89 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 970.10 | 926.21 | 920.34 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 15:15:00 | 924.95 | 936.65 | 937.10 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 951.50 | 939.89 | 938.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 963.75 | 944.66 | 940.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 14:15:00 | 971.65 | 972.50 | 962.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 14:45:00 | 972.55 | 972.50 | 962.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 1001.00 | 1001.26 | 994.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:30:00 | 1010.45 | 1001.26 | 994.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 997.85 | 1000.58 | 994.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:45:00 | 996.75 | 1000.58 | 994.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 1005.90 | 1001.43 | 996.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:30:00 | 1010.50 | 999.84 | 997.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:15:00 | 1007.70 | 999.84 | 997.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 11:45:00 | 1007.60 | 1001.87 | 999.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:45:00 | 1008.00 | 1002.89 | 999.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 978.95 | 999.80 | 999.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 978.95 | 999.80 | 999.58 | SL hit (close<static) qty=1.00 sl=996.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 968.90 | 993.62 | 996.79 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1063.55 | 1001.69 | 994.75 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 1024.00 | 1046.88 | 1048.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 1013.30 | 1033.49 | 1041.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 12:15:00 | 1034.80 | 1032.24 | 1039.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:30:00 | 1032.40 | 1032.24 | 1039.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1061.10 | 1038.24 | 1039.73 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1068.10 | 1044.21 | 1042.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1093.20 | 1067.69 | 1056.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 1081.90 | 1084.96 | 1073.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:30:00 | 1082.30 | 1084.96 | 1073.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 1082.70 | 1084.18 | 1075.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 1077.20 | 1084.18 | 1075.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 1082.00 | 1083.06 | 1076.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 1077.40 | 1083.06 | 1076.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1077.20 | 1082.20 | 1077.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 1076.30 | 1082.20 | 1077.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1075.70 | 1080.90 | 1076.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:15:00 | 1072.00 | 1080.90 | 1076.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1070.30 | 1078.78 | 1076.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 1070.30 | 1078.78 | 1076.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1073.60 | 1077.74 | 1076.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 1066.20 | 1077.74 | 1076.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1074.00 | 1076.46 | 1075.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 1074.00 | 1076.46 | 1075.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 1072.20 | 1075.61 | 1075.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 1071.20 | 1075.61 | 1075.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1075.00 | 1075.98 | 1075.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 1075.00 | 1075.98 | 1075.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1076.50 | 1076.08 | 1075.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:30:00 | 1073.90 | 1076.08 | 1075.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 1074.00 | 1075.67 | 1075.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:00:00 | 1074.00 | 1075.67 | 1075.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 1073.30 | 1075.19 | 1075.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 15:15:00 | 1068.70 | 1073.61 | 1074.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1057.00 | 1051.50 | 1057.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1057.00 | 1051.50 | 1057.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1057.00 | 1051.50 | 1057.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1057.00 | 1051.50 | 1057.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1055.90 | 1052.38 | 1057.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 1046.60 | 1051.23 | 1056.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 1049.30 | 1050.25 | 1054.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:00:00 | 1050.50 | 1050.76 | 1053.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:45:00 | 1051.30 | 1051.47 | 1053.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1040.80 | 1049.33 | 1052.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 1035.70 | 1045.83 | 1050.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:45:00 | 1036.20 | 1045.14 | 1046.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 11:00:00 | 1035.40 | 1043.19 | 1045.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 1030.90 | 1040.73 | 1044.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1043.70 | 1038.66 | 1042.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 1043.70 | 1038.66 | 1042.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1040.90 | 1039.11 | 1042.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 1052.20 | 1039.11 | 1042.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1055.70 | 1042.43 | 1043.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1055.70 | 1042.43 | 1043.39 | SL hit (close>static) qty=1.00 sl=1054.90 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 1048.10 | 1044.73 | 1044.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 12:15:00 | 1051.30 | 1046.04 | 1044.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 1045.10 | 1046.37 | 1045.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 1045.10 | 1046.37 | 1045.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1045.10 | 1046.37 | 1045.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 1047.30 | 1046.37 | 1045.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1045.50 | 1046.20 | 1045.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 1038.00 | 1046.20 | 1045.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 1036.30 | 1044.22 | 1044.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 1030.90 | 1041.56 | 1043.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 1042.40 | 1040.72 | 1042.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 1042.40 | 1040.72 | 1042.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1042.40 | 1040.72 | 1042.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 1042.40 | 1040.72 | 1042.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1029.50 | 1038.48 | 1041.37 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 1044.50 | 1041.98 | 1041.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 1053.50 | 1044.80 | 1043.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1061.10 | 1062.72 | 1054.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:45:00 | 1062.00 | 1062.72 | 1054.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1057.20 | 1060.40 | 1055.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 12:15:00 | 1059.00 | 1060.40 | 1055.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 1044.20 | 1057.16 | 1054.03 | SL hit (close<static) qty=1.00 sl=1052.50 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 1101.70 | 1114.63 | 1115.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1094.00 | 1108.56 | 1112.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1067.50 | 1065.42 | 1079.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 1067.50 | 1065.42 | 1079.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1074.70 | 1067.30 | 1074.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 1076.90 | 1067.30 | 1074.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1072.80 | 1068.40 | 1074.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 1071.60 | 1068.44 | 1073.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 1059.50 | 1051.03 | 1050.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 1059.50 | 1051.03 | 1050.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1070.60 | 1055.93 | 1052.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 1057.80 | 1063.43 | 1059.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 10:15:00 | 1057.80 | 1063.43 | 1059.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1057.80 | 1063.43 | 1059.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 1057.80 | 1063.43 | 1059.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 1065.10 | 1063.77 | 1060.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 1079.70 | 1066.81 | 1061.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 1086.60 | 1102.49 | 1102.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 1086.60 | 1102.49 | 1102.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 1082.40 | 1098.47 | 1100.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 14:15:00 | 1085.50 | 1084.86 | 1089.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:45:00 | 1088.10 | 1084.86 | 1089.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1087.20 | 1081.89 | 1083.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 1089.00 | 1081.89 | 1083.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1088.20 | 1083.15 | 1084.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 1082.10 | 1084.22 | 1084.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 1082.20 | 1083.16 | 1084.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 12:45:00 | 1085.10 | 1082.38 | 1083.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 1092.00 | 1084.31 | 1084.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 13:15:00 | 1092.00 | 1084.31 | 1084.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 1104.90 | 1092.40 | 1088.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 1091.20 | 1094.38 | 1091.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 1091.20 | 1094.38 | 1091.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1091.20 | 1094.38 | 1091.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 1091.20 | 1094.38 | 1091.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1081.40 | 1091.79 | 1090.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 1081.40 | 1091.79 | 1090.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 1077.50 | 1088.93 | 1089.21 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 1086.90 | 1083.73 | 1083.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1100.30 | 1091.23 | 1087.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 1095.80 | 1102.78 | 1098.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 1095.80 | 1102.78 | 1098.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1095.80 | 1102.78 | 1098.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 1100.90 | 1102.78 | 1098.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1095.60 | 1101.34 | 1098.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 1095.60 | 1101.34 | 1098.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 1094.00 | 1099.88 | 1097.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:30:00 | 1086.70 | 1099.88 | 1097.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1094.60 | 1096.80 | 1096.80 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 12:15:00 | 1098.40 | 1097.04 | 1096.90 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1093.00 | 1096.23 | 1096.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 1085.00 | 1093.98 | 1095.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 14:15:00 | 1083.80 | 1077.33 | 1084.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 14:15:00 | 1083.80 | 1077.33 | 1084.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1083.80 | 1077.33 | 1084.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1083.80 | 1077.33 | 1084.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1086.00 | 1079.07 | 1084.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1106.50 | 1079.07 | 1084.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1111.70 | 1085.59 | 1086.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1106.30 | 1085.59 | 1086.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 1091.90 | 1088.26 | 1088.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 13:15:00 | 1096.70 | 1090.90 | 1089.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 1085.80 | 1089.88 | 1089.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 14:15:00 | 1085.80 | 1089.88 | 1089.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1085.80 | 1089.88 | 1089.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1085.80 | 1089.88 | 1089.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1083.30 | 1088.56 | 1088.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1048.20 | 1088.56 | 1088.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 1062.50 | 1083.35 | 1086.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1032.50 | 1047.74 | 1059.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 1029.00 | 1021.59 | 1037.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 1029.00 | 1021.59 | 1037.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 780.80 | 769.17 | 776.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:45:00 | 780.30 | 769.17 | 776.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 777.00 | 770.74 | 776.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:30:00 | 781.50 | 770.74 | 776.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 776.75 | 771.60 | 775.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 776.75 | 771.60 | 775.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 776.65 | 772.61 | 775.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 776.65 | 772.61 | 775.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 772.95 | 772.68 | 775.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 770.70 | 773.36 | 774.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:00:00 | 770.30 | 772.75 | 774.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:30:00 | 770.95 | 772.05 | 773.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 770.85 | 772.05 | 773.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 771.35 | 770.43 | 772.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 771.35 | 770.43 | 772.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 770.10 | 770.36 | 772.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:15:00 | 769.00 | 770.36 | 772.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 768.80 | 770.21 | 771.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 15:15:00 | 773.50 | 771.59 | 772.00 | SL hit (close>static) qty=1.00 sl=772.95 alert=retest2 |

### Cycle 85 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 780.75 | 772.45 | 771.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 785.15 | 776.21 | 773.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 807.75 | 810.70 | 800.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 807.75 | 810.70 | 800.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 803.75 | 808.93 | 805.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 803.75 | 808.93 | 805.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 803.50 | 807.85 | 805.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 804.05 | 807.85 | 805.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 799.20 | 805.20 | 804.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 798.35 | 805.20 | 804.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 797.50 | 803.66 | 803.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 794.55 | 799.22 | 800.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 761.55 | 758.83 | 765.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:45:00 | 763.15 | 758.83 | 765.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 778.60 | 762.97 | 766.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 778.60 | 762.97 | 766.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 787.25 | 767.83 | 768.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 787.25 | 767.83 | 768.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 789.50 | 772.16 | 770.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 792.00 | 786.42 | 780.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 785.85 | 787.99 | 783.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:45:00 | 786.35 | 787.99 | 783.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 783.40 | 787.07 | 783.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 783.75 | 787.07 | 783.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 783.25 | 786.31 | 783.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 781.40 | 786.31 | 783.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 778.80 | 784.80 | 782.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 778.80 | 784.80 | 782.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 782.40 | 784.32 | 782.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 785.55 | 784.00 | 782.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 783.85 | 783.66 | 782.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 783.80 | 782.46 | 782.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-18 15:15:00 | 864.11 | 850.30 | 842.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 865.15 | 884.39 | 885.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 860.25 | 868.21 | 873.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 10:15:00 | 878.20 | 868.79 | 871.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 10:15:00 | 878.20 | 868.79 | 871.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 878.20 | 868.79 | 871.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 878.20 | 868.79 | 871.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 882.65 | 871.56 | 872.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 882.65 | 871.56 | 872.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 886.00 | 874.45 | 874.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 890.65 | 877.69 | 875.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 892.85 | 893.02 | 888.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 893.15 | 893.02 | 888.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 889.40 | 892.30 | 888.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:15:00 | 887.80 | 892.30 | 888.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 883.00 | 890.44 | 888.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 880.45 | 890.44 | 888.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 885.90 | 889.53 | 887.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 887.85 | 888.86 | 887.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:00:00 | 889.20 | 888.54 | 887.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 878.10 | 885.92 | 886.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 878.10 | 885.92 | 886.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 876.00 | 881.94 | 884.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 852.90 | 852.79 | 860.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 15:00:00 | 852.90 | 852.79 | 860.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 850.65 | 845.97 | 850.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:00:00 | 843.00 | 847.65 | 849.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 843.85 | 843.49 | 845.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 858.20 | 844.14 | 844.76 | SL hit (close>static) qty=1.00 sl=856.60 alert=retest2 |

### Cycle 91 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 864.90 | 848.29 | 846.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 875.30 | 853.69 | 849.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 917.90 | 921.03 | 908.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 917.90 | 921.03 | 908.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 917.90 | 921.03 | 908.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 934.80 | 922.08 | 910.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 932.95 | 922.08 | 910.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 934.00 | 927.26 | 916.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 930.10 | 935.95 | 928.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 935.60 | 935.88 | 929.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:45:00 | 936.20 | 935.60 | 929.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 937.10 | 935.60 | 929.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 922.80 | 931.12 | 929.95 | SL hit (close<static) qty=1.00 sl=925.95 alert=retest2 |

### Cycle 92 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 929.55 | 933.90 | 934.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 915.60 | 930.24 | 932.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 901.95 | 899.27 | 907.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:00:00 | 901.95 | 899.27 | 907.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 905.00 | 900.41 | 907.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 905.05 | 900.41 | 907.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 904.25 | 901.18 | 907.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:00:00 | 904.25 | 901.18 | 907.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 898.60 | 901.12 | 906.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:30:00 | 904.75 | 901.12 | 906.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 900.85 | 896.05 | 900.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 900.85 | 896.05 | 900.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 896.05 | 896.05 | 899.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 893.05 | 896.05 | 899.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 899.95 | 896.83 | 899.74 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 910.65 | 902.15 | 901.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 915.25 | 905.75 | 903.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 15:15:00 | 914.40 | 914.57 | 911.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 09:15:00 | 914.90 | 914.57 | 911.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 914.10 | 915.36 | 912.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:45:00 | 915.55 | 915.39 | 912.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 910.20 | 921.07 | 919.61 | SL hit (close<static) qty=1.00 sl=911.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 907.65 | 918.39 | 918.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 905.00 | 913.97 | 916.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 912.75 | 911.51 | 914.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:00:00 | 912.75 | 911.51 | 914.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 912.60 | 911.73 | 914.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:30:00 | 912.00 | 911.73 | 914.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 884.50 | 878.74 | 884.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 884.50 | 878.74 | 884.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 881.00 | 879.19 | 884.43 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 906.65 | 888.09 | 887.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 912.00 | 892.88 | 889.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 910.00 | 910.16 | 904.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:15:00 | 905.50 | 910.16 | 904.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 910.50 | 910.23 | 904.99 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 899.55 | 907.75 | 908.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 15:15:00 | 897.95 | 904.69 | 906.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 887.00 | 884.52 | 893.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:15:00 | 888.55 | 884.52 | 893.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 890.00 | 885.61 | 892.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 891.05 | 885.61 | 892.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 890.00 | 886.49 | 892.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 892.45 | 886.49 | 892.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 900.00 | 888.36 | 890.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 900.00 | 888.36 | 890.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 895.95 | 889.88 | 891.20 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 896.65 | 892.06 | 892.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 13:15:00 | 903.00 | 894.25 | 893.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 895.45 | 897.37 | 895.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 895.45 | 897.37 | 895.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 895.45 | 897.37 | 895.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 895.45 | 897.37 | 895.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 889.70 | 895.83 | 894.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 889.70 | 895.83 | 894.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 884.15 | 893.50 | 893.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 880.45 | 890.89 | 892.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 882.95 | 879.13 | 884.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:15:00 | 883.55 | 879.13 | 884.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 884.65 | 880.23 | 884.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 885.50 | 880.23 | 884.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 883.70 | 880.93 | 884.08 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 889.40 | 884.42 | 884.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 904.45 | 888.43 | 886.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 915.80 | 919.51 | 907.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:45:00 | 918.05 | 919.51 | 907.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 906.20 | 916.85 | 907.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 906.20 | 916.85 | 907.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 912.55 | 915.99 | 907.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 915.00 | 915.51 | 908.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 916.10 | 915.97 | 913.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 907.20 | 911.33 | 911.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 907.20 | 911.33 | 911.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 894.30 | 907.92 | 909.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 898.40 | 898.21 | 903.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:00:00 | 898.40 | 898.21 | 903.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 903.25 | 899.22 | 903.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 903.25 | 899.22 | 903.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 904.90 | 900.36 | 903.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 906.25 | 900.36 | 903.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 898.90 | 900.06 | 903.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 893.85 | 900.06 | 903.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 895.15 | 899.13 | 901.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 915.00 | 902.93 | 902.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 915.00 | 902.93 | 902.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 929.35 | 908.21 | 905.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 932.65 | 933.31 | 925.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:30:00 | 931.95 | 933.31 | 925.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 932.65 | 934.42 | 927.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 930.50 | 934.42 | 927.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 945.25 | 936.12 | 929.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 958.20 | 936.12 | 929.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 935.55 | 942.96 | 943.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 935.55 | 942.96 | 943.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 931.65 | 940.70 | 942.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 938.00 | 937.28 | 940.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:15:00 | 946.55 | 937.28 | 940.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 941.75 | 938.17 | 940.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 944.60 | 938.17 | 940.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 943.85 | 939.31 | 940.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:30:00 | 941.05 | 939.62 | 940.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:00:00 | 940.55 | 939.80 | 940.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 948.25 | 941.49 | 941.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 948.25 | 941.49 | 941.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 949.00 | 945.46 | 943.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 998.70 | 999.53 | 988.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 998.70 | 999.53 | 988.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 995.00 | 1002.41 | 997.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 995.00 | 1002.41 | 997.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 996.00 | 1001.13 | 997.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 993.95 | 1001.13 | 997.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 997.15 | 1000.33 | 997.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 996.80 | 1000.33 | 997.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1002.60 | 1000.79 | 997.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 13:30:00 | 1005.20 | 1000.96 | 998.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:15:00 | 1004.45 | 1000.96 | 998.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 15:15:00 | 1006.00 | 1001.55 | 998.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 1005.80 | 1002.79 | 999.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 991.00 | 1000.43 | 998.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 991.00 | 1000.43 | 998.98 | SL hit (close<static) qty=1.00 sl=995.65 alert=retest2 |

### Cycle 104 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 987.25 | 996.02 | 997.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 981.30 | 993.08 | 995.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 977.20 | 974.40 | 981.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 977.20 | 974.40 | 981.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 975.70 | 973.74 | 979.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 982.15 | 973.74 | 979.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 979.90 | 974.97 | 979.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 983.60 | 974.97 | 979.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 974.75 | 974.93 | 978.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 966.00 | 973.36 | 977.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 968.05 | 972.30 | 976.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 985.05 | 976.77 | 975.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 985.05 | 976.77 | 975.97 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 972.45 | 975.29 | 975.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 969.35 | 974.10 | 974.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 12:15:00 | 960.00 | 959.55 | 963.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-20 12:30:00 | 958.80 | 959.55 | 963.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 842.40 | 831.16 | 841.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:45:00 | 842.95 | 831.16 | 841.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 838.90 | 832.70 | 840.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:15:00 | 844.15 | 832.70 | 840.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 845.30 | 835.22 | 841.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:45:00 | 847.35 | 835.22 | 841.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 844.35 | 837.05 | 841.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 851.45 | 837.05 | 841.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 846.85 | 839.99 | 842.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 846.10 | 839.99 | 842.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 846.50 | 841.29 | 842.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:30:00 | 846.70 | 841.29 | 842.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 848.65 | 843.57 | 843.43 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 14:15:00 | 841.45 | 843.15 | 843.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 15:15:00 | 837.90 | 842.10 | 842.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 798.35 | 797.87 | 810.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 796.30 | 797.87 | 810.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 830.30 | 806.45 | 811.61 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 821.00 | 815.05 | 814.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 843.00 | 822.18 | 818.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 840.80 | 840.92 | 832.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:45:00 | 843.75 | 840.92 | 832.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 835.10 | 841.72 | 836.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 834.95 | 841.72 | 836.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 835.55 | 840.49 | 836.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 838.90 | 840.49 | 836.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 838.45 | 840.08 | 836.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 15:00:00 | 850.60 | 841.72 | 838.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 843.90 | 852.37 | 852.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 843.90 | 852.37 | 852.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 836.00 | 848.19 | 850.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 845.25 | 841.29 | 844.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 845.25 | 841.29 | 844.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 845.25 | 841.29 | 844.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 845.25 | 841.29 | 844.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 845.30 | 842.09 | 844.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:45:00 | 846.50 | 842.09 | 844.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 842.50 | 842.17 | 844.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:45:00 | 842.30 | 842.85 | 844.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 13:15:00 | 850.00 | 844.28 | 845.16 | SL hit (close>static) qty=1.00 sl=847.60 alert=retest2 |

### Cycle 111 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 853.15 | 846.05 | 845.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 10:15:00 | 857.25 | 849.61 | 847.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 857.45 | 862.86 | 859.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 857.45 | 862.86 | 859.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 857.45 | 862.86 | 859.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 857.45 | 862.86 | 859.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 852.00 | 860.69 | 858.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 852.00 | 860.69 | 858.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 846.70 | 855.60 | 856.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 843.80 | 853.24 | 855.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 854.90 | 851.95 | 853.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 854.90 | 851.95 | 853.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 854.90 | 851.95 | 853.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 854.90 | 851.95 | 853.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 851.60 | 851.88 | 853.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 856.10 | 851.88 | 853.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 854.05 | 852.31 | 853.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 854.05 | 852.31 | 853.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 848.35 | 851.52 | 853.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 845.05 | 850.14 | 852.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 844.50 | 848.52 | 851.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 851.80 | 841.87 | 841.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 851.80 | 841.87 | 841.72 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 834.00 | 845.04 | 845.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 824.90 | 836.34 | 840.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 788.80 | 786.06 | 797.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 788.80 | 786.06 | 797.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 796.90 | 788.75 | 796.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 796.15 | 788.75 | 796.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 782.75 | 787.55 | 795.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 778.35 | 783.27 | 790.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 739.43 | 772.74 | 784.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 755.55 | 754.10 | 769.19 | SL hit (close>ema200) qty=0.50 sl=754.10 alert=retest2 |

### Cycle 115 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 789.20 | 775.90 | 774.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 797.00 | 780.12 | 776.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 783.85 | 788.77 | 783.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 783.85 | 788.77 | 783.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 775.35 | 786.08 | 782.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 775.35 | 786.08 | 782.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 776.00 | 784.07 | 781.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 763.35 | 784.07 | 781.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 759.05 | 779.06 | 779.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 751.90 | 762.47 | 768.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 766.90 | 762.31 | 767.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 766.90 | 762.31 | 767.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 766.90 | 762.31 | 767.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 770.45 | 762.31 | 767.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 767.75 | 763.39 | 767.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 769.30 | 763.39 | 767.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 782.20 | 767.16 | 768.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 782.20 | 767.16 | 768.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 781.20 | 769.96 | 769.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 773.80 | 769.96 | 769.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 774.55 | 770.88 | 770.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 774.55 | 770.88 | 770.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 782.30 | 775.03 | 772.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 784.40 | 793.57 | 786.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 784.40 | 793.57 | 786.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 784.40 | 793.57 | 786.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 783.45 | 793.57 | 786.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 786.70 | 792.19 | 786.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 795.00 | 792.75 | 786.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 808.75 | 789.87 | 787.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 759.25 | 788.51 | 789.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 759.25 | 788.51 | 789.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 749.40 | 780.69 | 786.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 763.45 | 760.45 | 771.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 763.45 | 760.45 | 771.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 763.45 | 760.45 | 771.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 759.05 | 761.73 | 770.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 783.05 | 767.75 | 772.05 | SL hit (close>static) qty=1.00 sl=782.60 alert=retest2 |

### Cycle 119 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 784.00 | 776.05 | 775.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 806.70 | 782.18 | 778.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 788.60 | 796.44 | 789.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 788.60 | 796.44 | 789.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 788.60 | 796.44 | 789.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 788.60 | 796.44 | 789.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 794.45 | 796.04 | 790.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 797.40 | 795.15 | 790.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:15:00 | 799.90 | 795.34 | 791.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 773.55 | 791.71 | 790.59 | SL hit (close<static) qty=1.00 sl=788.10 alert=retest2 |

### Cycle 120 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 769.30 | 787.23 | 788.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 760.70 | 777.18 | 783.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 792.30 | 774.60 | 780.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 792.30 | 774.60 | 780.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 792.30 | 774.60 | 780.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 792.30 | 774.60 | 780.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 780.70 | 775.82 | 780.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 794.20 | 775.82 | 780.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 793.90 | 782.30 | 782.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:45:00 | 793.90 | 782.30 | 782.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 787.30 | 783.30 | 782.94 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 778.55 | 782.35 | 782.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 757.55 | 777.18 | 780.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 776.30 | 771.58 | 775.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 776.30 | 771.58 | 775.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 776.30 | 771.58 | 775.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:45:00 | 775.35 | 771.58 | 775.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 785.15 | 774.30 | 776.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 785.15 | 774.30 | 776.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 779.65 | 775.37 | 776.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 781.40 | 775.37 | 776.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 790.95 | 778.48 | 778.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 795.70 | 781.93 | 779.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 857.70 | 860.13 | 838.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 857.70 | 860.13 | 838.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 850.95 | 864.26 | 858.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 856.85 | 862.09 | 858.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 856.15 | 861.02 | 857.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 847.70 | 854.72 | 855.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 847.70 | 854.72 | 855.55 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 865.95 | 856.31 | 856.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 894.90 | 868.71 | 862.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 12:15:00 | 916.95 | 917.73 | 905.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 13:00:00 | 916.95 | 917.73 | 905.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 907.95 | 914.77 | 906.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:15:00 | 906.55 | 914.77 | 906.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 906.55 | 913.12 | 906.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 997.40 | 913.12 | 906.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 1036.80 | 1042.47 | 1043.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 15:15:00 | 1036.80 | 1042.47 | 1043.23 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1059.60 | 1045.90 | 1044.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1068.20 | 1051.21 | 1047.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1047.00 | 1051.32 | 1048.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 1047.00 | 1051.32 | 1048.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1047.00 | 1051.32 | 1048.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 1047.00 | 1051.32 | 1048.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1057.10 | 1052.48 | 1049.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:00:00 | 1059.10 | 1053.80 | 1050.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 14:00:00 | 1059.00 | 1055.45 | 1051.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 1072.80 | 1053.54 | 1051.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 747.50 | 2024-05-17 13:15:00 | 736.80 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest1 | 2024-05-27 10:45:00 | 799.00 | 2024-05-28 09:15:00 | 790.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest1 | 2024-05-27 12:00:00 | 800.90 | 2024-05-28 09:15:00 | 790.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest1 | 2024-05-27 15:00:00 | 800.00 | 2024-05-28 09:15:00 | 790.70 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-05-28 13:15:00 | 797.05 | 2024-05-28 14:15:00 | 789.25 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-06-03 11:00:00 | 726.60 | 2024-06-04 10:15:00 | 653.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-03 13:30:00 | 730.20 | 2024-06-04 10:15:00 | 657.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-03 14:45:00 | 729.60 | 2024-06-04 10:15:00 | 656.64 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 703.55 | 2024-06-04 10:15:00 | 668.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:45:00 | 687.05 | 2024-06-04 11:15:00 | 652.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 703.55 | 2024-06-04 12:15:00 | 633.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-04 10:45:00 | 687.05 | 2024-06-04 12:15:00 | 618.35 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-07-03 09:15:00 | 810.30 | 2024-07-05 14:15:00 | 793.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-07-03 10:15:00 | 804.00 | 2024-07-05 14:15:00 | 793.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-07-03 11:45:00 | 802.20 | 2024-07-05 14:15:00 | 793.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-07-03 14:30:00 | 803.15 | 2024-07-05 14:15:00 | 793.10 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-07-04 14:30:00 | 809.30 | 2024-07-05 14:15:00 | 793.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-07-09 13:00:00 | 790.20 | 2024-07-10 14:15:00 | 809.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-07-10 09:45:00 | 789.00 | 2024-07-10 14:15:00 | 809.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-07-10 13:15:00 | 790.70 | 2024-07-10 14:15:00 | 809.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-07-23 12:15:00 | 749.05 | 2024-07-26 09:15:00 | 799.45 | STOP_HIT | 1.00 | -6.73% |
| SELL | retest2 | 2024-07-24 14:00:00 | 778.40 | 2024-07-26 09:15:00 | 799.45 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-07-24 15:00:00 | 773.90 | 2024-07-26 09:15:00 | 799.45 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-08-02 13:30:00 | 817.60 | 2024-08-05 09:15:00 | 788.20 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2024-08-02 15:15:00 | 811.50 | 2024-08-05 09:15:00 | 788.20 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-08-19 09:15:00 | 829.00 | 2024-08-19 12:15:00 | 821.40 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-08-19 10:30:00 | 826.10 | 2024-08-19 12:15:00 | 821.40 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-08-19 11:45:00 | 826.15 | 2024-08-19 12:15:00 | 821.40 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-09-04 12:15:00 | 1046.55 | 2024-09-13 09:15:00 | 1151.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-05 12:45:00 | 1039.55 | 2024-09-13 09:15:00 | 1143.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-05 13:15:00 | 1034.70 | 2024-09-13 09:15:00 | 1138.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-06 10:45:00 | 1043.35 | 2024-09-13 09:15:00 | 1147.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-11 10:15:00 | 1105.75 | 2024-09-16 13:15:00 | 1067.10 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-09-16 11:15:00 | 1092.85 | 2024-09-16 13:15:00 | 1067.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-09-19 09:45:00 | 1049.85 | 2024-09-24 14:15:00 | 997.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 09:45:00 | 1049.85 | 2024-09-25 14:15:00 | 1000.25 | STOP_HIT | 0.50 | 4.72% |
| BUY | retest2 | 2024-10-15 14:15:00 | 975.70 | 2024-10-16 09:15:00 | 969.30 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-10-15 15:00:00 | 980.75 | 2024-10-16 09:15:00 | 969.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-10-21 14:00:00 | 918.90 | 2024-10-22 14:15:00 | 872.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 918.90 | 2024-10-23 09:15:00 | 914.45 | STOP_HIT | 0.50 | 0.48% |
| BUY | retest2 | 2024-10-25 09:15:00 | 988.25 | 2024-10-28 11:15:00 | 928.25 | STOP_HIT | 1.00 | -6.07% |
| BUY | retest2 | 2024-10-28 10:00:00 | 943.45 | 2024-10-28 11:15:00 | 928.25 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-11-05 14:15:00 | 968.10 | 2024-11-08 09:15:00 | 962.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-11-26 10:15:00 | 851.05 | 2024-11-27 12:15:00 | 869.25 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-11-29 14:30:00 | 885.60 | 2024-12-02 10:15:00 | 873.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-12-02 13:45:00 | 885.05 | 2024-12-09 09:15:00 | 973.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 09:15:00 | 886.05 | 2024-12-09 09:15:00 | 974.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-23 10:15:00 | 859.45 | 2024-12-27 10:15:00 | 858.15 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2024-12-23 10:45:00 | 859.15 | 2024-12-27 10:15:00 | 858.15 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-01-10 14:00:00 | 845.00 | 2025-01-13 11:15:00 | 802.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 15:15:00 | 851.00 | 2025-01-13 11:15:00 | 808.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 14:00:00 | 845.00 | 2025-01-14 09:15:00 | 823.60 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-01-10 15:15:00 | 851.00 | 2025-01-14 09:15:00 | 823.60 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-01-15 12:30:00 | 851.30 | 2025-01-15 13:15:00 | 850.00 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-01-31 10:15:00 | 879.80 | 2025-02-04 11:15:00 | 862.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-01-31 12:45:00 | 878.00 | 2025-02-04 11:15:00 | 862.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-01-31 13:15:00 | 877.25 | 2025-02-04 11:15:00 | 862.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-01-31 15:15:00 | 885.10 | 2025-02-04 11:15:00 | 862.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-02-13 13:15:00 | 818.00 | 2025-02-17 09:15:00 | 777.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 818.00 | 2025-02-18 10:15:00 | 783.55 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-02-28 09:30:00 | 763.45 | 2025-03-04 11:15:00 | 791.90 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-03-17 09:15:00 | 818.25 | 2025-03-17 13:15:00 | 801.50 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-03-21 09:15:00 | 839.95 | 2025-04-01 12:15:00 | 857.35 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2025-04-24 09:30:00 | 1010.50 | 2025-04-25 09:15:00 | 978.95 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-04-24 10:15:00 | 1007.70 | 2025-04-25 09:15:00 | 978.95 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-04-24 11:45:00 | 1007.60 | 2025-04-25 09:15:00 | 978.95 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-04-24 12:45:00 | 1008.00 | 2025-04-25 09:15:00 | 978.95 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-05-21 12:00:00 | 1046.60 | 2025-05-28 09:15:00 | 1055.70 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-05-21 15:00:00 | 1049.30 | 2025-05-28 09:15:00 | 1055.70 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-05-22 11:00:00 | 1050.50 | 2025-05-28 09:15:00 | 1055.70 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-05-22 11:45:00 | 1051.30 | 2025-05-28 09:15:00 | 1055.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-05-22 13:30:00 | 1035.70 | 2025-05-28 11:15:00 | 1048.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-05-27 09:45:00 | 1036.20 | 2025-05-28 11:15:00 | 1048.10 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-05-27 11:00:00 | 1035.40 | 2025-05-28 11:15:00 | 1048.10 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-05-27 12:00:00 | 1030.90 | 2025-05-28 11:15:00 | 1048.10 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-06-03 12:15:00 | 1059.00 | 2025-06-03 12:15:00 | 1044.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-06-04 10:30:00 | 1059.30 | 2025-06-12 11:15:00 | 1101.70 | STOP_HIT | 1.00 | 4.00% |
| BUY | retest2 | 2025-06-04 13:00:00 | 1058.80 | 2025-06-12 11:15:00 | 1101.70 | STOP_HIT | 1.00 | 4.05% |
| BUY | retest2 | 2025-06-04 14:30:00 | 1058.70 | 2025-06-12 11:15:00 | 1101.70 | STOP_HIT | 1.00 | 4.06% |
| BUY | retest2 | 2025-06-05 13:45:00 | 1061.40 | 2025-06-12 11:15:00 | 1101.70 | STOP_HIT | 1.00 | 3.80% |
| BUY | retest2 | 2025-06-05 14:30:00 | 1062.10 | 2025-06-12 11:15:00 | 1101.70 | STOP_HIT | 1.00 | 3.73% |
| BUY | retest2 | 2025-06-06 09:30:00 | 1075.10 | 2025-06-12 11:15:00 | 1101.70 | STOP_HIT | 1.00 | 2.47% |
| SELL | retest2 | 2025-06-17 11:30:00 | 1071.60 | 2025-06-23 14:15:00 | 1059.50 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2025-06-25 12:45:00 | 1079.70 | 2025-07-02 09:15:00 | 1086.60 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-07-07 15:15:00 | 1082.10 | 2025-07-08 13:15:00 | 1092.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-08 09:45:00 | 1082.20 | 2025-07-08 13:15:00 | 1092.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-08 12:45:00 | 1085.10 | 2025-07-08 13:15:00 | 1092.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-08-12 09:15:00 | 770.70 | 2025-08-13 15:15:00 | 773.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-08-12 10:00:00 | 770.30 | 2025-08-13 15:15:00 | 773.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-08-12 13:30:00 | 770.95 | 2025-08-14 11:15:00 | 773.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-08-12 14:00:00 | 770.85 | 2025-08-18 10:15:00 | 780.75 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-13 11:15:00 | 769.00 | 2025-08-18 10:15:00 | 780.75 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-08-13 11:45:00 | 768.80 | 2025-08-18 10:15:00 | 780.75 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-14 10:00:00 | 769.80 | 2025-08-18 10:15:00 | 780.75 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-08-14 13:15:00 | 769.25 | 2025-08-18 10:15:00 | 780.75 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-08-14 14:15:00 | 766.90 | 2025-08-18 10:15:00 | 780.75 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-09-05 09:15:00 | 785.55 | 2025-09-18 15:15:00 | 864.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 10:15:00 | 783.85 | 2025-09-18 15:15:00 | 862.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 13:15:00 | 783.80 | 2025-09-18 15:15:00 | 862.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-07 12:45:00 | 887.85 | 2025-10-08 10:15:00 | 878.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-07 15:00:00 | 889.20 | 2025-10-08 10:15:00 | 878.10 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-10-16 12:00:00 | 843.00 | 2025-10-20 10:15:00 | 858.20 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-10-17 11:45:00 | 843.85 | 2025-10-20 10:15:00 | 858.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-10-28 11:30:00 | 934.80 | 2025-10-31 10:15:00 | 922.80 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-28 12:15:00 | 932.95 | 2025-10-31 10:15:00 | 922.80 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-28 15:00:00 | 934.00 | 2025-11-04 15:15:00 | 929.55 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-10-30 10:15:00 | 930.10 | 2025-11-04 15:15:00 | 929.55 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-10-30 11:45:00 | 936.20 | 2025-11-04 15:15:00 | 929.55 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-30 12:15:00 | 937.10 | 2025-11-04 15:15:00 | 929.55 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-03 09:30:00 | 937.40 | 2025-11-04 15:15:00 | 929.55 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-04 09:45:00 | 937.40 | 2025-11-04 15:15:00 | 929.55 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-17 12:45:00 | 915.55 | 2025-11-19 10:15:00 | 910.20 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-15 13:15:00 | 915.00 | 2025-12-17 11:15:00 | 907.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-16 14:45:00 | 916.10 | 2025-12-17 11:15:00 | 907.20 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-18 14:15:00 | 893.85 | 2025-12-19 13:15:00 | 915.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-12-19 11:15:00 | 895.15 | 2025-12-19 13:15:00 | 915.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-12-24 10:15:00 | 958.20 | 2025-12-29 11:15:00 | 935.55 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-12-30 11:30:00 | 941.05 | 2025-12-30 13:15:00 | 948.25 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-30 13:00:00 | 940.55 | 2025-12-30 13:15:00 | 948.25 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-01-07 13:30:00 | 1005.20 | 2026-01-08 10:15:00 | 991.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-01-07 14:15:00 | 1004.45 | 2026-01-08 10:15:00 | 991.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-01-07 15:15:00 | 1006.00 | 2026-01-08 10:15:00 | 991.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-01-08 09:30:00 | 1005.80 | 2026-01-08 10:15:00 | 991.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-01-13 10:30:00 | 966.00 | 2026-01-16 09:15:00 | 985.05 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-01-13 12:00:00 | 968.05 | 2026-01-16 09:15:00 | 985.05 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-02-06 15:00:00 | 850.60 | 2026-02-12 09:15:00 | 843.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-02-16 12:45:00 | 842.30 | 2026-02-16 13:15:00 | 850.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-02-23 10:30:00 | 845.05 | 2026-02-25 11:15:00 | 851.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-23 11:45:00 | 844.50 | 2026-02-25 11:15:00 | 851.80 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-03-06 14:30:00 | 778.35 | 2026-03-09 09:15:00 | 739.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 778.35 | 2026-03-09 14:15:00 | 755.55 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2026-03-10 13:00:00 | 781.00 | 2026-03-10 14:15:00 | 789.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-03-17 11:15:00 | 773.80 | 2026-03-17 11:15:00 | 774.55 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-03-19 12:00:00 | 795.00 | 2026-03-23 09:15:00 | 759.25 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2026-03-20 09:15:00 | 808.75 | 2026-03-23 09:15:00 | 759.25 | STOP_HIT | 1.00 | -6.12% |
| SELL | retest2 | 2026-03-24 10:30:00 | 759.05 | 2026-03-24 12:15:00 | 783.05 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2026-03-27 13:15:00 | 797.40 | 2026-03-30 09:15:00 | 773.55 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2026-03-27 15:15:00 | 799.90 | 2026-03-30 09:15:00 | 773.55 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-04-13 10:45:00 | 856.85 | 2026-04-13 14:15:00 | 847.70 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-04-13 11:45:00 | 856.15 | 2026-04-13 14:15:00 | 847.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-04-21 09:15:00 | 997.40 | 2026-05-05 15:15:00 | 1036.80 | STOP_HIT | 1.00 | 3.95% |
