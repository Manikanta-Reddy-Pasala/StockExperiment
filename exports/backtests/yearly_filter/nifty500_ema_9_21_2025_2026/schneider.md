# Schneider Electric Infrastructure Ltd. (SCHNEIDER)

## Backtest Summary

- **Window:** 2024-08-23 09:15:00 → 2026-05-11 15:15:00 (2938 bars)
- **Last close:** 1316.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 54 |
| ALERT1 | 41 |
| ALERT2 | 41 |
| ALERT2_SKIP | 24 |
| ALERT3 | 340 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 595.85 | 569.17 | 565.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 602.10 | 575.75 | 569.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 659.85 | 661.88 | 651.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:15:00 | 659.85 | 661.88 | 651.85 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 654.10 | 660.60 | 654.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:15:00 | 654.10 | 660.60 | 654.56 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 654.10 | 659.30 | 654.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:15:00 | 654.10 | 659.30 | 654.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 656.90 | 658.82 | 654.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:15:00 | 656.90 | 658.82 | 654.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 669.35 | 660.92 | 656.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 669.35 | 660.92 | 656.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 670.00 | 666.12 | 661.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 670.00 | 666.12 | 661.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 670.50 | 669.10 | 665.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:15:00 | 670.50 | 669.10 | 665.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 667.05 | 669.96 | 667.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:15:00 | 667.05 | 669.96 | 667.69 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 667.00 | 669.37 | 667.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:15:00 | 667.00 | 669.37 | 667.62 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 719.10 | 684.84 | 676.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 719.10 | 684.84 | 676.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 750.60 | 753.31 | 745.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:15:00 | 750.60 | 753.31 | 745.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 761.55 | 754.43 | 747.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 761.55 | 754.43 | 747.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 763.15 | 760.87 | 757.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 763.15 | 760.87 | 757.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 765.00 | 768.42 | 765.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:15:00 | 765.00 | 768.42 | 765.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 766.90 | 768.11 | 765.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 766.90 | 768.11 | 765.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 781.40 | 770.77 | 767.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:15:00 | 781.40 | 770.77 | 767.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 786.90 | 790.25 | 784.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:15:00 | 786.90 | 790.25 | 784.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 787.45 | 789.69 | 784.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 787.45 | 789.69 | 784.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 780.95 | 787.94 | 784.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:15:00 | 780.95 | 787.94 | 784.43 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 779.00 | 786.15 | 783.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:15:00 | 779.00 | 786.15 | 783.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 771.05 | 781.59 | 782.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 763.30 | 775.24 | 778.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 730.35 | 729.67 | 742.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:15:00 | 730.35 | 729.67 | 742.77 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 739.50 | 731.63 | 742.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:15:00 | 739.50 | 731.63 | 742.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 753.50 | 737.44 | 742.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:15:00 | 753.50 | 737.44 | 742.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 749.00 | 739.76 | 743.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:15:00 | 749.00 | 739.76 | 743.14 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 769.90 | 745.78 | 745.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 12:15:00 | 786.00 | 771.97 | 762.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 773.40 | 777.37 | 768.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 773.40 | 777.37 | 768.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 773.40 | 777.37 | 768.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 773.40 | 777.37 | 768.67 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 764.50 | 774.79 | 768.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:15:00 | 764.50 | 774.79 | 768.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 765.60 | 772.96 | 768.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:15:00 | 765.60 | 772.96 | 768.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 764.90 | 771.34 | 767.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:15:00 | 764.90 | 771.34 | 767.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 760.00 | 769.08 | 767.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:15:00 | 760.00 | 769.08 | 767.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 751.00 | 763.13 | 764.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 749.65 | 759.61 | 762.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 760.50 | 759.69 | 761.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 760.50 | 759.69 | 761.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 760.50 | 759.69 | 761.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 760.50 | 759.69 | 761.67 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 762.00 | 760.15 | 761.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:15:00 | 762.00 | 760.15 | 761.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 764.40 | 761.00 | 761.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:15:00 | 764.40 | 761.00 | 761.94 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 768.90 | 762.58 | 762.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 773.00 | 765.34 | 763.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 798.00 | 798.83 | 791.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:15:00 | 798.00 | 798.83 | 791.22 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 796.45 | 800.36 | 795.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:15:00 | 796.45 | 800.36 | 795.71 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 802.00 | 800.69 | 796.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:15:00 | 802.00 | 800.69 | 796.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 832.40 | 807.03 | 799.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 832.40 | 807.03 | 799.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 826.00 | 832.52 | 825.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 826.00 | 832.52 | 825.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 824.95 | 831.01 | 825.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:15:00 | 824.95 | 831.01 | 825.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 826.50 | 830.10 | 825.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:15:00 | 826.50 | 830.10 | 825.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 826.20 | 829.32 | 825.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:15:00 | 826.20 | 829.32 | 825.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 826.75 | 828.81 | 825.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:15:00 | 826.75 | 828.81 | 825.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 825.00 | 828.05 | 825.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:15:00 | 825.00 | 828.05 | 825.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 822.50 | 826.94 | 825.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:15:00 | 822.50 | 826.94 | 825.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 825.00 | 826.72 | 825.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:15:00 | 825.00 | 826.72 | 825.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 828.90 | 827.16 | 825.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:15:00 | 828.90 | 827.16 | 825.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 827.00 | 827.12 | 825.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:15:00 | 827.00 | 827.12 | 825.90 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 821.30 | 825.96 | 825.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:15:00 | 821.30 | 825.96 | 825.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 820.35 | 824.84 | 825.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 817.80 | 822.64 | 823.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 827.00 | 821.06 | 822.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 827.00 | 821.06 | 822.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 827.00 | 821.06 | 822.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 827.00 | 821.06 | 822.24 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 817.05 | 820.25 | 821.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:15:00 | 817.05 | 820.25 | 821.77 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 822.95 | 820.79 | 821.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:15:00 | 822.95 | 820.79 | 821.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 817.95 | 820.22 | 821.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:15:00 | 817.95 | 820.22 | 821.52 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 824.00 | 819.39 | 820.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:15:00 | 824.00 | 819.39 | 820.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 816.05 | 818.72 | 820.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 816.05 | 818.72 | 820.21 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 820.00 | 818.98 | 820.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:15:00 | 820.00 | 818.98 | 820.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 816.10 | 818.40 | 819.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:15:00 | 816.10 | 818.40 | 819.82 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 814.95 | 817.71 | 819.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:15:00 | 814.95 | 817.71 | 819.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 819.00 | 817.06 | 818.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:15:00 | 819.00 | 817.06 | 818.73 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 814.85 | 816.61 | 818.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:15:00 | 814.85 | 816.61 | 818.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 815.30 | 816.35 | 818.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 815.30 | 816.35 | 818.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 826.00 | 818.28 | 818.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:15:00 | 826.00 | 818.28 | 818.82 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 826.00 | 819.83 | 819.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 829.95 | 824.52 | 822.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 843.90 | 843.94 | 837.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 843.90 | 843.94 | 837.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 843.90 | 843.94 | 837.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 843.90 | 843.94 | 837.36 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 840.00 | 844.90 | 840.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:15:00 | 840.00 | 844.90 | 840.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 841.00 | 844.12 | 840.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:15:00 | 841.00 | 844.12 | 840.16 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 843.95 | 844.08 | 840.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:15:00 | 843.95 | 844.08 | 840.50 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 848.00 | 844.87 | 841.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 848.00 | 844.87 | 841.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 903.60 | 898.21 | 884.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 903.60 | 898.21 | 884.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 923.00 | 914.12 | 906.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 923.00 | 914.12 | 906.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 927.95 | 933.22 | 925.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 927.95 | 933.22 | 925.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 916.00 | 929.78 | 924.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:15:00 | 916.00 | 929.78 | 924.34 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 923.45 | 928.51 | 924.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:15:00 | 923.45 | 928.51 | 924.26 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 926.00 | 928.01 | 924.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:15:00 | 926.00 | 928.01 | 924.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 920.65 | 926.54 | 924.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:15:00 | 920.65 | 926.54 | 924.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 930.00 | 927.23 | 924.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:15:00 | 930.00 | 927.23 | 924.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 913.75 | 924.20 | 923.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 913.75 | 924.20 | 923.67 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 908.70 | 921.10 | 922.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 882.00 | 907.34 | 914.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 898.90 | 898.52 | 907.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:15:00 | 898.90 | 898.52 | 907.51 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 901.00 | 894.56 | 900.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:15:00 | 901.00 | 894.56 | 900.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 909.95 | 897.64 | 901.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:15:00 | 909.95 | 897.64 | 901.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 909.70 | 900.05 | 902.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:15:00 | 909.70 | 900.05 | 902.28 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 925.50 | 905.14 | 904.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 941.25 | 916.18 | 909.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 12:15:00 | 989.90 | 1004.24 | 978.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 12:15:00 | 989.90 | 1004.24 | 978.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 989.90 | 1004.24 | 978.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:15:00 | 989.90 | 1004.24 | 978.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 998.00 | 1002.99 | 980.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:15:00 | 998.00 | 1002.99 | 980.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 997.00 | 1001.47 | 983.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:15:00 | 997.00 | 1001.47 | 983.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1012.00 | 1003.58 | 986.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 1012.00 | 1003.58 | 986.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1013.50 | 1023.83 | 1015.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:15:00 | 1013.50 | 1023.83 | 1015.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1009.30 | 1020.92 | 1014.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:15:00 | 1009.30 | 1020.92 | 1014.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1006.90 | 1018.12 | 1013.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:15:00 | 1006.90 | 1018.12 | 1013.90 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1000.00 | 1014.49 | 1012.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:15:00 | 1000.00 | 1014.49 | 1012.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 986.30 | 1006.78 | 1009.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 13:15:00 | 951.50 | 985.20 | 997.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 1007.50 | 987.71 | 996.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 15:15:00 | 1007.50 | 987.71 | 996.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1007.50 | 987.71 | 996.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:15:00 | 1007.50 | 987.71 | 996.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 882.60 | 869.60 | 885.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:15:00 | 882.60 | 869.60 | 885.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 886.30 | 872.94 | 885.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:15:00 | 886.30 | 872.94 | 885.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 886.30 | 875.62 | 885.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:15:00 | 886.30 | 875.62 | 885.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 869.10 | 868.82 | 875.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 869.10 | 868.82 | 875.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 870.20 | 857.52 | 865.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:15:00 | 870.20 | 857.52 | 865.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 867.30 | 859.47 | 866.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:15:00 | 867.30 | 859.47 | 866.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 859.80 | 859.54 | 865.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 859.80 | 859.54 | 865.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 857.00 | 853.69 | 858.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 857.00 | 853.69 | 858.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 845.00 | 851.95 | 857.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:15:00 | 845.00 | 851.95 | 857.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 854.00 | 847.49 | 852.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 854.00 | 847.49 | 852.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 862.00 | 850.39 | 853.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:15:00 | 862.00 | 850.39 | 853.39 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 860.10 | 852.33 | 854.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:15:00 | 860.10 | 852.33 | 854.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 14:15:00 | 860.00 | 855.59 | 855.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 863.10 | 857.51 | 856.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 13:15:00 | 860.20 | 860.79 | 858.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:15:00 | 860.20 | 860.79 | 858.52 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 860.00 | 860.63 | 858.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:15:00 | 860.00 | 860.63 | 858.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 861.90 | 860.89 | 858.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:15:00 | 861.90 | 860.89 | 858.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 860.00 | 860.71 | 859.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 860.00 | 860.71 | 859.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 857.80 | 860.13 | 858.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:15:00 | 857.80 | 860.13 | 858.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 860.00 | 860.10 | 859.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:15:00 | 860.00 | 860.10 | 859.03 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 867.50 | 861.58 | 859.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:15:00 | 867.50 | 861.58 | 859.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 860.00 | 861.22 | 859.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:15:00 | 860.00 | 861.22 | 859.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 858.00 | 860.58 | 859.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:15:00 | 858.00 | 860.58 | 859.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 861.80 | 860.82 | 859.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 861.80 | 860.82 | 859.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 860.00 | 860.66 | 859.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:15:00 | 860.00 | 860.66 | 859.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 860.00 | 860.53 | 859.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:15:00 | 860.00 | 860.53 | 859.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 860.00 | 860.42 | 859.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:15:00 | 860.00 | 860.42 | 859.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 860.00 | 860.34 | 859.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:15:00 | 860.00 | 860.34 | 859.97 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 860.00 | 860.17 | 859.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 860.00 | 860.17 | 859.98 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 866.50 | 861.43 | 860.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:15:00 | 866.50 | 861.43 | 860.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 863.50 | 862.41 | 861.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:15:00 | 863.50 | 862.41 | 861.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 864.00 | 862.73 | 861.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:15:00 | 864.00 | 862.73 | 861.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 865.00 | 863.18 | 861.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:15:00 | 865.00 | 863.18 | 861.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 855.50 | 861.65 | 861.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 855.50 | 861.65 | 861.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 858.70 | 861.06 | 860.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:15:00 | 858.70 | 861.06 | 860.98 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 856.00 | 860.05 | 860.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 09:15:00 | 854.00 | 858.16 | 859.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 861.00 | 851.24 | 854.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 861.00 | 851.24 | 854.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 861.00 | 851.24 | 854.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 861.00 | 851.24 | 854.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 873.80 | 855.75 | 855.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:15:00 | 873.80 | 855.75 | 855.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 875.20 | 859.64 | 857.67 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 858.00 | 863.22 | 863.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 850.00 | 858.94 | 861.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 876.00 | 860.77 | 861.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 876.00 | 860.77 | 861.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 876.00 | 860.77 | 861.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 876.00 | 860.77 | 861.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 880.05 | 864.62 | 863.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 11:15:00 | 886.00 | 868.90 | 865.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 13:15:00 | 880.00 | 886.69 | 879.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:15:00 | 880.00 | 886.69 | 879.52 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 877.95 | 884.94 | 879.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:15:00 | 877.95 | 884.94 | 879.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 874.85 | 882.92 | 878.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:15:00 | 874.85 | 882.92 | 878.96 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 860.00 | 875.45 | 876.08 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 881.45 | 874.35 | 874.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 886.70 | 876.82 | 875.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 874.00 | 880.08 | 877.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 874.00 | 880.08 | 877.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 874.00 | 880.08 | 877.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:15:00 | 874.00 | 880.08 | 877.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 877.45 | 879.55 | 877.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:15:00 | 877.45 | 879.55 | 877.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 876.75 | 878.99 | 877.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:15:00 | 876.75 | 878.99 | 877.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 874.00 | 877.99 | 877.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:15:00 | 874.00 | 877.99 | 877.16 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 866.80 | 875.76 | 876.22 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 887.05 | 877.58 | 876.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 903.70 | 889.21 | 883.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 908.30 | 909.00 | 901.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 908.30 | 909.00 | 901.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 908.30 | 909.00 | 901.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 908.30 | 909.00 | 901.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 902.60 | 907.72 | 901.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:15:00 | 902.60 | 907.72 | 901.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 899.10 | 906.00 | 901.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:15:00 | 899.10 | 906.00 | 901.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 895.00 | 903.80 | 900.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:15:00 | 895.00 | 903.80 | 900.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 901.35 | 903.31 | 901.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:15:00 | 901.35 | 903.31 | 901.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 901.50 | 902.95 | 901.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:15:00 | 901.50 | 902.95 | 901.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 900.45 | 902.45 | 901.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:15:00 | 900.45 | 902.45 | 901.01 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 898.60 | 901.68 | 900.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 898.60 | 901.68 | 900.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 893.35 | 900.01 | 900.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 15:15:00 | 890.00 | 896.41 | 898.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 886.00 | 885.91 | 889.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 886.00 | 885.91 | 889.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 886.00 | 885.91 | 889.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:15:00 | 886.00 | 885.91 | 889.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 844.95 | 839.94 | 846.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:15:00 | 844.95 | 839.94 | 846.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 832.15 | 823.87 | 830.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:15:00 | 832.15 | 823.87 | 830.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 829.00 | 824.90 | 830.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:15:00 | 829.00 | 824.90 | 830.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 827.95 | 825.80 | 830.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:15:00 | 827.95 | 825.80 | 830.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 844.05 | 829.45 | 831.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 844.05 | 829.45 | 831.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 850.80 | 833.72 | 833.11 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 838.00 | 841.80 | 842.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 832.30 | 839.90 | 841.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 13:15:00 | 837.20 | 836.87 | 839.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 13:15:00 | 837.20 | 836.87 | 839.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 837.20 | 836.87 | 839.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:15:00 | 837.20 | 836.87 | 839.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 833.40 | 836.17 | 838.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:15:00 | 833.40 | 836.17 | 838.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 824.95 | 833.55 | 837.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 824.95 | 833.55 | 837.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 827.45 | 828.02 | 832.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:15:00 | 827.45 | 828.02 | 832.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 836.55 | 829.28 | 832.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 836.55 | 829.28 | 832.33 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 842.95 | 832.01 | 833.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:15:00 | 842.95 | 832.01 | 833.29 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 831.50 | 831.91 | 833.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:15:00 | 831.50 | 831.91 | 833.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 826.40 | 829.10 | 831.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 826.40 | 829.10 | 831.09 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 833.25 | 829.48 | 830.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:15:00 | 833.25 | 829.48 | 830.71 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 833.00 | 830.19 | 830.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:15:00 | 833.00 | 830.19 | 830.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 836.85 | 832.29 | 831.80 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 826.30 | 831.25 | 831.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 817.35 | 828.47 | 830.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 831.10 | 822.58 | 825.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 831.10 | 822.58 | 825.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 831.10 | 822.58 | 825.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 831.10 | 822.58 | 825.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 826.50 | 823.37 | 825.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:15:00 | 826.50 | 823.37 | 825.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 825.15 | 823.72 | 825.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:15:00 | 825.15 | 823.72 | 825.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 822.40 | 823.46 | 825.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:15:00 | 822.40 | 823.46 | 825.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 829.05 | 824.42 | 825.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:15:00 | 829.05 | 824.42 | 825.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 829.75 | 825.49 | 825.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:15:00 | 829.75 | 825.49 | 825.94 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 832.65 | 826.92 | 826.55 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 823.30 | 826.55 | 826.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 813.00 | 823.33 | 825.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 824.45 | 819.23 | 821.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 824.45 | 819.23 | 821.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 824.45 | 819.23 | 821.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:15:00 | 824.45 | 819.23 | 821.65 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 820.40 | 819.46 | 821.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:15:00 | 820.40 | 819.46 | 821.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 827.40 | 821.05 | 822.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 827.40 | 821.05 | 822.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 816.85 | 821.42 | 822.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:15:00 | 816.85 | 821.42 | 822.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 830.00 | 819.45 | 820.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 830.00 | 819.45 | 820.55 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 10:15:00 | 840.50 | 823.66 | 822.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 859.55 | 839.37 | 831.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 857.05 | 862.11 | 855.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 857.05 | 862.11 | 855.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 857.05 | 862.11 | 855.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 857.05 | 862.11 | 855.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 850.10 | 859.71 | 854.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 850.10 | 859.71 | 854.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 852.00 | 858.17 | 854.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:15:00 | 852.00 | 858.17 | 854.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 857.55 | 857.21 | 854.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:15:00 | 857.55 | 857.21 | 854.64 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 854.15 | 856.60 | 854.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:15:00 | 854.15 | 856.60 | 854.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 853.15 | 855.91 | 854.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:15:00 | 853.15 | 855.91 | 854.47 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 868.40 | 858.41 | 855.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 868.40 | 858.41 | 855.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 857.00 | 858.06 | 856.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:15:00 | 857.00 | 858.06 | 856.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 851.85 | 856.82 | 855.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:15:00 | 851.85 | 856.82 | 855.65 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 852.55 | 855.96 | 855.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:15:00 | 852.55 | 855.96 | 855.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 851.40 | 854.45 | 854.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 841.70 | 851.90 | 853.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 849.65 | 843.65 | 847.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 849.65 | 843.65 | 847.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 849.65 | 843.65 | 847.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 849.65 | 843.65 | 847.78 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 857.00 | 846.32 | 848.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:15:00 | 857.00 | 846.32 | 848.62 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 856.90 | 848.44 | 849.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:15:00 | 856.90 | 848.44 | 849.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 856.40 | 850.03 | 850.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 14:15:00 | 862.85 | 853.50 | 851.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 873.00 | 882.34 | 872.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:15:00 | 873.00 | 882.34 | 872.41 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 874.70 | 880.81 | 872.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:15:00 | 874.70 | 880.81 | 872.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 872.45 | 879.14 | 872.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:15:00 | 872.45 | 879.14 | 872.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 875.10 | 878.33 | 872.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:15:00 | 875.10 | 878.33 | 872.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 875.65 | 877.79 | 873.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:15:00 | 875.65 | 877.79 | 873.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 876.00 | 877.41 | 873.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:15:00 | 876.00 | 877.41 | 873.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 858.85 | 873.70 | 872.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 858.85 | 873.70 | 872.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 845.35 | 868.03 | 869.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 808.00 | 850.81 | 860.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 820.90 | 808.16 | 822.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 14:15:00 | 820.90 | 808.16 | 822.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 820.90 | 808.16 | 822.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:15:00 | 820.90 | 808.16 | 822.42 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 821.30 | 810.79 | 822.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:15:00 | 821.30 | 810.79 | 822.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 820.15 | 813.66 | 821.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:15:00 | 820.15 | 813.66 | 821.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 817.50 | 814.43 | 821.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:15:00 | 817.50 | 814.43 | 821.31 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 820.00 | 815.54 | 821.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:15:00 | 820.00 | 815.54 | 821.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 819.90 | 816.42 | 821.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:15:00 | 819.90 | 816.42 | 821.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 823.00 | 817.73 | 821.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:15:00 | 823.00 | 817.73 | 821.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 821.00 | 818.39 | 821.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:15:00 | 821.00 | 818.39 | 821.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 808.65 | 816.44 | 820.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 808.65 | 816.44 | 820.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 781.05 | 771.26 | 778.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:15:00 | 781.05 | 771.26 | 778.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 779.00 | 772.80 | 778.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:15:00 | 779.00 | 772.80 | 778.71 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 781.00 | 774.44 | 778.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:15:00 | 781.00 | 774.44 | 778.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 782.20 | 775.99 | 779.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:15:00 | 782.20 | 775.99 | 779.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 729.45 | 722.11 | 731.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 729.45 | 722.11 | 731.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 727.40 | 723.17 | 731.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:15:00 | 727.40 | 723.17 | 731.01 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 730.30 | 725.46 | 730.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:15:00 | 730.30 | 725.46 | 730.21 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 733.75 | 727.12 | 730.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:15:00 | 733.75 | 727.12 | 730.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 733.20 | 728.34 | 730.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:15:00 | 733.20 | 728.34 | 730.77 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 722.60 | 720.84 | 724.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:15:00 | 722.60 | 720.84 | 724.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 717.20 | 720.11 | 723.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:15:00 | 717.20 | 720.11 | 723.46 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 723.00 | 718.99 | 721.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 723.00 | 718.99 | 721.42 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 723.70 | 719.93 | 721.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:15:00 | 723.70 | 719.93 | 721.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 719.10 | 719.77 | 721.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:15:00 | 719.10 | 719.77 | 721.40 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 720.90 | 719.19 | 720.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:15:00 | 720.90 | 719.19 | 720.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 722.50 | 719.85 | 720.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 722.50 | 719.85 | 720.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 722.45 | 720.37 | 720.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:15:00 | 722.45 | 720.37 | 720.88 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 11:15:00 | 735.65 | 723.43 | 722.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 13:15:00 | 740.30 | 727.85 | 724.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 10:15:00 | 759.75 | 762.55 | 750.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:15:00 | 759.75 | 762.55 | 750.96 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 756.00 | 760.45 | 752.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:15:00 | 756.00 | 760.45 | 752.82 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 764.95 | 761.35 | 753.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:15:00 | 764.95 | 761.35 | 753.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 759.50 | 761.55 | 755.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 759.50 | 761.55 | 755.34 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 750.70 | 761.09 | 757.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:15:00 | 750.70 | 761.09 | 757.34 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 753.25 | 759.52 | 756.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:15:00 | 753.25 | 759.52 | 756.97 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 750.50 | 758.58 | 757.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 750.50 | 758.58 | 757.04 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 750.95 | 757.06 | 756.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:15:00 | 750.95 | 757.06 | 756.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 751.80 | 756.01 | 756.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 749.00 | 754.60 | 755.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 742.25 | 742.17 | 747.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:15:00 | 742.25 | 742.17 | 747.37 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 750.75 | 743.76 | 747.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:15:00 | 750.75 | 743.76 | 747.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 747.55 | 744.52 | 747.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:15:00 | 747.55 | 744.52 | 747.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 741.35 | 743.89 | 746.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 741.35 | 743.89 | 746.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 719.00 | 717.32 | 720.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 719.00 | 717.32 | 720.93 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 717.50 | 717.35 | 720.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:15:00 | 717.50 | 717.35 | 720.62 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 710.45 | 710.52 | 713.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 710.45 | 710.52 | 713.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 693.75 | 697.55 | 704.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 693.75 | 697.55 | 704.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 700.45 | 697.62 | 701.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:15:00 | 700.45 | 697.62 | 701.68 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 710.15 | 700.47 | 702.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 710.15 | 700.47 | 702.29 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 710.00 | 702.37 | 702.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:15:00 | 710.00 | 702.37 | 702.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 709.35 | 703.77 | 703.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 714.10 | 705.83 | 704.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 736.60 | 737.97 | 728.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:15:00 | 736.60 | 737.97 | 728.54 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 738.00 | 743.14 | 738.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:15:00 | 738.00 | 743.14 | 738.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 739.00 | 742.31 | 738.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:15:00 | 739.00 | 742.31 | 738.64 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 734.95 | 740.84 | 738.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 734.95 | 740.84 | 738.31 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 732.05 | 739.08 | 737.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:15:00 | 732.05 | 739.08 | 737.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 732.70 | 736.54 | 736.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 727.10 | 734.65 | 735.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 711.20 | 708.60 | 715.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 711.20 | 708.60 | 715.95 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 722.00 | 711.28 | 716.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 722.00 | 711.28 | 716.50 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 721.85 | 713.40 | 716.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:15:00 | 721.85 | 713.40 | 716.98 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 724.20 | 718.10 | 718.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:15:00 | 724.20 | 718.10 | 718.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 722.90 | 719.06 | 718.88 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 715.75 | 718.36 | 718.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 712.35 | 716.38 | 717.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 716.05 | 715.97 | 717.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 716.05 | 715.97 | 717.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 716.05 | 715.97 | 717.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 716.05 | 715.97 | 717.05 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 721.30 | 717.03 | 717.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:15:00 | 721.30 | 717.03 | 717.44 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 715.05 | 716.64 | 717.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:15:00 | 715.05 | 716.64 | 717.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 719.50 | 717.21 | 717.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:15:00 | 719.50 | 717.21 | 717.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 716.80 | 717.13 | 717.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:15:00 | 716.80 | 717.13 | 717.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 716.95 | 717.09 | 717.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:15:00 | 716.95 | 717.09 | 717.33 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 714.50 | 716.57 | 717.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:15:00 | 714.50 | 716.57 | 717.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 703.00 | 705.63 | 710.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 703.00 | 705.63 | 710.02 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 696.20 | 698.35 | 703.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 696.20 | 698.35 | 703.47 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 701.60 | 698.43 | 702.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:15:00 | 701.60 | 698.43 | 702.16 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 703.90 | 699.52 | 702.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:15:00 | 703.90 | 699.52 | 702.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 712.35 | 702.09 | 703.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:15:00 | 712.35 | 702.09 | 703.23 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 710.00 | 703.67 | 703.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:15:00 | 710.00 | 703.67 | 703.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 699.20 | 702.79 | 703.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:15:00 | 699.20 | 702.79 | 703.42 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 627.00 | 622.62 | 627.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 627.00 | 622.62 | 627.30 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 622.20 | 622.54 | 626.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:15:00 | 622.20 | 622.54 | 626.83 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 611.35 | 588.87 | 596.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 611.35 | 588.87 | 596.16 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 633.60 | 597.82 | 599.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:15:00 | 633.60 | 597.82 | 599.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 623.10 | 602.87 | 601.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 661.65 | 634.68 | 624.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 678.45 | 678.57 | 657.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:15:00 | 678.45 | 678.57 | 657.99 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 680.90 | 689.88 | 681.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 680.90 | 689.88 | 681.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 685.00 | 688.90 | 682.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:15:00 | 685.00 | 688.90 | 682.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 684.40 | 688.00 | 682.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:15:00 | 684.40 | 688.00 | 682.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 741.40 | 739.53 | 728.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:15:00 | 741.40 | 739.53 | 728.15 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 732.85 | 738.69 | 730.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:15:00 | 732.85 | 738.69 | 730.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 765.00 | 742.94 | 733.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 765.00 | 742.94 | 733.92 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 790.00 | 798.96 | 790.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:15:00 | 790.00 | 798.96 | 790.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 790.00 | 797.17 | 790.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:15:00 | 790.00 | 797.17 | 790.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 814.25 | 800.73 | 793.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:15:00 | 814.25 | 800.73 | 793.24 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 798.40 | 805.69 | 800.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:15:00 | 798.40 | 805.69 | 800.50 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 814.50 | 807.46 | 801.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:15:00 | 814.50 | 807.46 | 801.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 899.00 | 915.41 | 905.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 899.00 | 915.41 | 905.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 908.60 | 914.05 | 905.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:15:00 | 908.60 | 914.05 | 905.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 914.30 | 914.10 | 906.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:15:00 | 914.30 | 914.10 | 906.31 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 914.90 | 913.51 | 907.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:15:00 | 914.90 | 913.51 | 907.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 870.00 | 904.73 | 904.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 870.00 | 904.73 | 904.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 885.00 | 900.78 | 902.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 868.75 | 894.38 | 899.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 896.00 | 887.71 | 894.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 896.00 | 887.71 | 894.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 896.00 | 887.71 | 894.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:15:00 | 896.00 | 887.71 | 894.59 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 892.05 | 888.58 | 894.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:15:00 | 892.05 | 888.58 | 894.36 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 872.90 | 880.73 | 887.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:15:00 | 872.90 | 880.73 | 887.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 887.80 | 879.71 | 885.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 887.80 | 879.71 | 885.41 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 888.00 | 881.37 | 885.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:15:00 | 888.00 | 881.37 | 885.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 885.00 | 882.10 | 885.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:15:00 | 885.00 | 882.10 | 885.58 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 893.00 | 884.28 | 886.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:15:00 | 893.00 | 884.28 | 886.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 891.00 | 885.62 | 886.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:15:00 | 891.00 | 885.62 | 886.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 899.60 | 888.42 | 887.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 15:15:00 | 900.25 | 890.78 | 888.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 885.00 | 896.47 | 893.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 885.00 | 896.47 | 893.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 885.00 | 896.47 | 893.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 885.00 | 896.47 | 893.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 890.00 | 895.18 | 893.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:15:00 | 890.00 | 895.18 | 893.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 875.00 | 891.14 | 891.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 863.10 | 885.53 | 889.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 872.00 | 852.89 | 864.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 872.00 | 852.89 | 864.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 872.00 | 852.89 | 864.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 872.00 | 852.89 | 864.23 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 874.35 | 857.18 | 865.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:15:00 | 874.35 | 857.18 | 865.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 874.35 | 868.72 | 868.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 909.00 | 876.78 | 872.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 879.00 | 896.37 | 887.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 879.00 | 896.37 | 887.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 879.00 | 896.37 | 887.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 879.00 | 896.37 | 887.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 897.40 | 892.11 | 887.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:15:00 | 897.40 | 892.11 | 887.37 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 900.00 | 893.69 | 888.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:15:00 | 900.00 | 893.69 | 888.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 908.95 | 915.42 | 906.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:15:00 | 908.95 | 915.42 | 906.89 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 899.00 | 912.14 | 906.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:15:00 | 899.00 | 912.14 | 906.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 898.95 | 909.50 | 905.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:15:00 | 898.95 | 909.50 | 905.52 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 895.00 | 905.24 | 904.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 895.00 | 905.24 | 904.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 901.75 | 904.97 | 904.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:15:00 | 901.75 | 904.97 | 904.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 899.95 | 903.97 | 904.18 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 905.00 | 904.34 | 904.32 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 882.20 | 900.41 | 902.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 862.90 | 888.81 | 896.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 874.00 | 868.45 | 879.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:15:00 | 874.00 | 868.45 | 879.81 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 888.00 | 873.41 | 880.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 888.00 | 873.41 | 880.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 882.00 | 875.13 | 880.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 882.00 | 875.13 | 880.32 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 882.80 | 876.66 | 880.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:15:00 | 882.80 | 876.66 | 880.55 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 876.35 | 876.60 | 880.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:15:00 | 876.35 | 876.60 | 880.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 875.00 | 876.28 | 879.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:15:00 | 875.00 | 876.28 | 879.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 892.50 | 879.52 | 880.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:15:00 | 892.50 | 879.52 | 880.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 889.10 | 881.44 | 881.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:15:00 | 889.10 | 881.44 | 881.61 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 905.10 | 886.17 | 883.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 910.00 | 899.05 | 894.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 12:15:00 | 894.00 | 900.67 | 896.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 894.00 | 900.67 | 896.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 894.00 | 900.67 | 896.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:15:00 | 894.00 | 900.67 | 896.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 900.00 | 900.54 | 897.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:15:00 | 900.00 | 900.54 | 897.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 892.95 | 899.02 | 896.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:15:00 | 892.95 | 899.02 | 896.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 890.00 | 897.21 | 896.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:15:00 | 890.00 | 897.21 | 896.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 851.45 | 888.06 | 892.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 846.10 | 874.21 | 884.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 878.40 | 872.14 | 881.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 13:15:00 | 878.40 | 872.14 | 881.79 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 870.00 | 871.71 | 880.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 14:15:00 | 870.00 | 871.71 | 880.71 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 875.00 | 870.49 | 878.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 875.00 | 870.49 | 878.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 876.45 | 871.20 | 877.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:15:00 | 876.45 | 871.20 | 877.39 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 875.00 | 871.96 | 877.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:15:00 | 875.00 | 871.96 | 877.17 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 870.20 | 871.61 | 876.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:15:00 | 870.20 | 871.61 | 876.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 897.55 | 876.80 | 878.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:15:00 | 897.55 | 876.80 | 878.45 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 895.35 | 880.51 | 879.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 906.70 | 885.74 | 882.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 892.50 | 894.59 | 888.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:15:00 | 892.50 | 894.59 | 888.93 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 884.10 | 892.24 | 888.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 884.10 | 892.24 | 888.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 879.90 | 889.77 | 888.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 879.90 | 889.77 | 888.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 876.00 | 887.02 | 886.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 876.00 | 887.02 | 886.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 873.00 | 884.21 | 885.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 868.50 | 881.07 | 884.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 888.05 | 859.86 | 866.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 888.05 | 859.86 | 866.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 888.05 | 859.86 | 866.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 888.05 | 859.86 | 866.95 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 870.00 | 869.55 | 870.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:15:00 | 870.00 | 869.55 | 870.20 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 871.00 | 869.84 | 870.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:15:00 | 871.00 | 869.84 | 870.27 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 882.80 | 872.43 | 871.41 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 850.00 | 867.95 | 869.46 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 895.95 | 871.76 | 870.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 907.90 | 878.99 | 873.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 923.00 | 928.09 | 914.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:15:00 | 923.00 | 928.09 | 914.85 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1003.00 | 996.34 | 979.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 1003.00 | 996.34 | 979.27 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 1008.00 | 1014.41 | 1002.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:15:00 | 1008.00 | 1014.41 | 1002.61 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 997.00 | 1010.22 | 1002.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:15:00 | 997.00 | 1010.22 | 1002.71 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 1000.00 | 1008.18 | 1002.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:15:00 | 1000.00 | 1008.18 | 1002.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 999.15 | 1003.76 | 1001.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:15:00 | 999.15 | 1003.76 | 1001.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 983.10 | 999.63 | 999.63 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1038.50 | 1004.79 | 1001.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 1039.50 | 1011.73 | 1004.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 13:15:00 | 1075.00 | 1077.45 | 1054.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 13:15:00 | 1075.00 | 1077.45 | 1054.42 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1060.00 | 1072.33 | 1055.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:15:00 | 1060.00 | 1072.33 | 1055.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1075.00 | 1072.87 | 1057.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1075.00 | 1072.87 | 1057.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1105.00 | 1121.43 | 1106.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1105.00 | 1121.43 | 1106.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1121.35 | 1122.47 | 1109.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:15:00 | 1121.35 | 1122.47 | 1109.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1170.00 | 1174.43 | 1155.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 1170.00 | 1174.43 | 1155.44 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1160.00 | 1170.83 | 1157.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:15:00 | 1160.00 | 1170.83 | 1157.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1152.90 | 1167.25 | 1156.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:15:00 | 1152.90 | 1167.25 | 1156.68 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1164.85 | 1166.77 | 1157.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:15:00 | 1164.85 | 1166.77 | 1157.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1138.50 | 1161.11 | 1155.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:15:00 | 1138.50 | 1161.11 | 1155.70 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1162.50 | 1161.39 | 1156.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 1162.50 | 1161.39 | 1156.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1149.95 | 1159.10 | 1155.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 1149.95 | 1159.10 | 1155.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1165.00 | 1160.28 | 1156.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:15:00 | 1165.00 | 1160.28 | 1156.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1182.00 | 1174.16 | 1165.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:15:00 | 1182.00 | 1174.16 | 1165.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1169.00 | 1175.50 | 1167.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1169.00 | 1175.50 | 1167.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1200.00 | 1180.40 | 1170.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:15:00 | 1200.00 | 1180.40 | 1170.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1189.00 | 1182.12 | 1172.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:15:00 | 1189.00 | 1182.12 | 1172.28 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1288.90 | 1324.90 | 1294.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 1288.90 | 1324.90 | 1294.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1300.00 | 1319.92 | 1295.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:15:00 | 1300.00 | 1319.92 | 1295.27 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1294.50 | 1314.83 | 1295.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:15:00 | 1294.50 | 1314.83 | 1295.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1295.00 | 1310.87 | 1295.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:15:00 | 1295.00 | 1310.87 | 1295.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1283.00 | 1305.29 | 1294.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:15:00 | 1283.00 | 1305.29 | 1294.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1282.50 | 1300.74 | 1293.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:15:00 | 1282.50 | 1300.74 | 1293.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1282.50 | 1297.09 | 1292.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:15:00 | 1282.50 | 1297.09 | 1292.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1293.00 | 1296.58 | 1292.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:15:00 | 1293.00 | 1296.58 | 1292.72 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 1313.00 | 1299.86 | 1294.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:15:00 | 1313.00 | 1299.86 | 1294.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1345.00 | 1327.92 | 1312.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 1345.00 | 1327.92 | 1312.08 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1345.00 | 1339.13 | 1324.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:15:00 | 1345.00 | 1339.13 | 1324.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 09:15:00 | 1309.00 | 1334.52 | 1325.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 09:15:00 | 1309.00 | 1334.52 | 1325.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 10:15:00 | 1300.00 | 1327.62 | 1322.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-11 10:15:00 | 1300.00 | 1327.62 | 1322.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-05-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-11 13:15:00 | 1288.50 | 1314.10 | 1317.32 | EMA200 below EMA400 |

