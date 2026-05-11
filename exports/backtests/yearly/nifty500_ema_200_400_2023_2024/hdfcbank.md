# HDFC Bank Ltd. (HDFCBANK)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 781.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 5 |
| ALERT3 | 57 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 3 |
| TARGET_HIT | 5 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 37
- **Target hits / Stop hits / Partials:** 5 / 40 / 3
- **Avg / median % per leg:** 0.51% / -0.73%
- **Sum % (uncompounded):** 24.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 5 | 16.1% | 3 | 28 | 0 | 0.01% | 0.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 5 | 16.1% | 3 | 28 | 0 | 0.01% | 0.4% |
| SELL (all) | 17 | 6 | 35.3% | 2 | 12 | 3 | 1.43% | 24.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 2 | 12 | 3 | 1.43% | 24.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 11 | 22.9% | 5 | 40 | 3 | 0.51% | 24.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 10:15:00 | 799.33 | 817.05 | 817.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 11:15:00 | 798.53 | 816.86 | 816.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 811.05 | 808.89 | 812.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 811.05 | 808.89 | 812.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 811.05 | 808.89 | 812.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:30:00 | 811.80 | 808.89 | 812.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 811.00 | 808.91 | 812.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:45:00 | 811.10 | 808.91 | 812.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 811.95 | 808.94 | 812.23 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 846.93 | 814.88 | 814.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 11:15:00 | 848.50 | 815.22 | 815.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 14:15:00 | 824.60 | 826.05 | 821.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-11 15:00:00 | 824.60 | 826.05 | 821.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 13:15:00 | 822.45 | 826.09 | 821.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 13:45:00 | 822.33 | 826.09 | 821.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 814.68 | 825.98 | 821.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 15:00:00 | 814.68 | 825.98 | 821.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 818.08 | 825.90 | 821.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 823.00 | 825.90 | 821.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 14:45:00 | 820.60 | 825.53 | 821.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 09:30:00 | 819.73 | 825.46 | 821.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 14:30:00 | 819.45 | 831.07 | 826.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 13:15:00 | 812.83 | 830.37 | 826.41 | SL hit (close<static) qty=1.00 sl=814.35 alert=retest2 |

### Cycle 3 — SELL (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 15:15:00 | 802.23 | 823.64 | 823.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 09:15:00 | 800.28 | 823.41 | 823.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 14:15:00 | 805.00 | 802.22 | 810.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-07 14:45:00 | 806.50 | 802.22 | 810.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 11:15:00 | 810.55 | 802.41 | 810.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 12:00:00 | 810.55 | 802.41 | 810.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 810.05 | 802.49 | 810.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 12:45:00 | 810.83 | 802.49 | 810.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 811.23 | 802.58 | 810.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 14:00:00 | 811.23 | 802.58 | 810.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 811.25 | 802.66 | 810.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 09:15:00 | 789.75 | 808.96 | 812.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 09:15:00 | 750.26 | 788.92 | 799.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 754.33 | 753.91 | 768.89 | SL hit (close>ema200) qty=0.50 sl=753.91 alert=retest2 |

### Cycle 4 — BUY (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 10:15:00 | 826.20 | 773.25 | 773.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 826.83 | 775.32 | 774.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 822.83 | 824.54 | 808.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-11 14:00:00 | 822.83 | 824.54 | 808.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 790.60 | 825.62 | 810.88 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 12:15:00 | 723.60 | 798.45 | 798.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 10:15:00 | 716.93 | 794.76 | 796.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 09:15:00 | 724.83 | 723.47 | 743.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 10:00:00 | 724.83 | 723.47 | 743.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 733.15 | 722.76 | 735.17 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 763.08 | 743.25 | 743.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 14:15:00 | 765.88 | 743.48 | 743.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 09:15:00 | 745.85 | 752.17 | 748.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 745.85 | 752.17 | 748.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 745.85 | 752.17 | 748.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 745.63 | 752.17 | 748.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 726.00 | 745.39 | 745.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 10:15:00 | 723.58 | 745.17 | 745.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 743.88 | 739.85 | 742.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 13:15:00 | 743.88 | 739.85 | 742.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 743.88 | 739.85 | 742.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 743.88 | 739.85 | 742.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 747.05 | 739.92 | 742.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 747.05 | 739.92 | 742.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 746.43 | 739.98 | 742.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 750.38 | 739.98 | 742.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 755.05 | 744.55 | 744.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 11:15:00 | 757.83 | 745.35 | 744.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 741.83 | 749.62 | 747.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 741.83 | 749.62 | 747.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 741.83 | 749.62 | 747.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 741.83 | 749.62 | 747.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 743.33 | 749.56 | 747.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 744.88 | 749.56 | 747.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 747.70 | 749.54 | 747.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:30:00 | 748.03 | 749.54 | 747.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 742.20 | 749.46 | 747.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 742.20 | 749.46 | 747.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 746.38 | 749.43 | 747.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 09:15:00 | 752.08 | 749.43 | 747.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 10:45:00 | 752.48 | 749.56 | 747.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 13:15:00 | 827.29 | 769.82 | 759.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 10:15:00 | 826.45 | 876.41 | 876.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 11:15:00 | 821.00 | 875.86 | 876.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 12:15:00 | 851.55 | 848.53 | 859.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 13:00:00 | 851.55 | 848.53 | 859.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 858.30 | 847.94 | 857.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:00:00 | 858.30 | 847.94 | 857.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 856.30 | 848.02 | 857.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:45:00 | 858.45 | 848.02 | 857.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 861.25 | 848.15 | 857.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:45:00 | 861.78 | 848.15 | 857.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 858.15 | 848.25 | 857.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 860.48 | 848.25 | 857.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 857.60 | 852.20 | 858.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 13:45:00 | 858.50 | 852.20 | 858.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 857.88 | 852.26 | 858.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:45:00 | 859.65 | 852.26 | 858.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 854.35 | 851.42 | 857.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 855.85 | 851.42 | 857.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 859.23 | 851.50 | 857.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 859.23 | 851.50 | 857.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 858.60 | 851.57 | 857.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 858.78 | 851.57 | 857.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 853.65 | 851.65 | 857.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 849.80 | 852.99 | 857.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:45:00 | 852.35 | 850.10 | 855.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 13:15:00 | 862.48 | 850.44 | 855.33 | SL hit (close>static) qty=1.00 sl=858.90 alert=retest2 |

### Cycle 10 — BUY (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 15:15:00 | 886.00 | 856.78 | 856.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 897.25 | 857.18 | 856.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 10:15:00 | 876.75 | 878.76 | 869.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-07 11:00:00 | 876.75 | 878.76 | 869.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 872.08 | 878.64 | 869.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 12:30:00 | 873.18 | 878.64 | 869.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 13:15:00 | 873.40 | 878.59 | 869.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:00:00 | 879.48 | 878.60 | 869.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 09:15:00 | 967.43 | 892.88 | 879.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 958.70 | 980.47 | 980.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 947.40 | 971.43 | 974.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 968.95 | 965.15 | 970.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 11:45:00 | 968.85 | 965.15 | 970.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 971.30 | 965.03 | 970.47 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 1000.55 | 974.00 | 973.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1008.40 | 974.35 | 974.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.98 | 981.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 15:00:00 | 984.05 | 986.98 | 981.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 979.85 | 986.89 | 981.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:30:00 | 976.80 | 986.89 | 981.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 981.75 | 986.84 | 981.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 983.00 | 986.84 | 981.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 976.50 | 986.75 | 981.90 | SL hit (close<static) qty=1.00 sl=979.10 alert=retest2 |

### Cycle 13 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.05 | 989.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.46 | 988.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 947.60 | 946.81 | 961.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 944.90 | 947.58 | 960.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:00:00 | 944.15 | 947.54 | 960.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 897.65 | 929.26 | 944.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 896.94 | 929.26 | 944.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 850.41 | 922.93 | 940.14 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-16 10:00:00 | 830.50 | 2023-05-19 09:15:00 | 818.30 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2023-05-18 09:15:00 | 825.50 | 2023-05-22 14:15:00 | 819.45 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-05-22 09:30:00 | 825.45 | 2023-05-24 12:15:00 | 811.10 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2023-07-13 09:15:00 | 823.00 | 2023-08-03 13:15:00 | 812.83 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-07-14 14:45:00 | 820.60 | 2023-08-03 13:15:00 | 812.83 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-07-17 09:30:00 | 819.73 | 2023-08-03 13:15:00 | 812.83 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-08-02 14:30:00 | 819.45 | 2023-08-03 13:15:00 | 812.83 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-09-20 09:15:00 | 789.75 | 2023-10-04 09:15:00 | 750.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-20 09:15:00 | 789.75 | 2023-11-15 09:15:00 | 754.33 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2023-12-06 09:30:00 | 809.45 | 2023-12-08 09:15:00 | 823.30 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2023-12-06 10:00:00 | 809.58 | 2023-12-08 09:15:00 | 823.30 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-06-05 09:15:00 | 752.08 | 2024-06-19 13:15:00 | 827.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 10:45:00 | 752.48 | 2024-06-19 13:15:00 | 827.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-20 09:15:00 | 849.80 | 2025-02-28 13:15:00 | 862.48 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-02-27 14:45:00 | 852.35 | 2025-02-28 13:15:00 | 862.48 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-03-03 11:15:00 | 851.30 | 2025-03-12 12:15:00 | 858.75 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-03-03 13:00:00 | 852.68 | 2025-03-12 12:15:00 | 858.75 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-03-05 13:30:00 | 845.95 | 2025-03-12 12:15:00 | 858.75 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-03-05 14:30:00 | 844.88 | 2025-03-12 12:15:00 | 858.75 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-03-07 11:15:00 | 846.45 | 2025-03-13 09:15:00 | 859.15 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-03-10 14:15:00 | 843.40 | 2025-03-13 09:15:00 | 859.15 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-03-13 14:30:00 | 850.63 | 2025-03-17 09:15:00 | 859.28 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-04-07 15:00:00 | 879.48 | 2025-04-21 09:15:00 | 967.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-06 11:15:00 | 983.00 | 2025-11-07 09:15:00 | 976.50 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-11-07 12:45:00 | 982.50 | 2025-11-11 10:15:00 | 979.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-11-07 13:15:00 | 983.65 | 2025-11-11 10:15:00 | 979.95 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-11-07 14:15:00 | 983.35 | 2025-11-14 09:15:00 | 980.05 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-11-10 10:15:00 | 990.05 | 2025-11-14 09:15:00 | 980.05 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-10 13:15:00 | 988.25 | 2025-12-17 10:15:00 | 987.20 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-11-11 13:15:00 | 987.95 | 2025-12-17 10:15:00 | 987.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-11-11 14:00:00 | 988.20 | 2025-12-22 10:15:00 | 987.30 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-11-14 15:00:00 | 990.45 | 2025-12-30 11:15:00 | 985.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-11-19 10:00:00 | 986.90 | 2025-12-31 09:15:00 | 989.60 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-11-19 11:45:00 | 986.65 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-12-10 13:15:00 | 989.70 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-12-10 15:15:00 | 991.70 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-12-11 10:00:00 | 994.40 | 2026-01-05 10:15:00 | 988.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-22 09:15:00 | 992.60 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-12-23 09:45:00 | 991.40 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-12-30 15:15:00 | 995.00 | 2026-01-05 13:15:00 | 978.15 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-31 12:30:00 | 993.40 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-12-31 15:15:00 | 994.00 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-01-02 09:15:00 | 994.60 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2026-01-02 15:00:00 | 1001.95 | 2026-01-06 09:15:00 | 963.25 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2026-02-06 09:15:00 | 944.90 | 2026-02-26 15:15:00 | 897.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 10:00:00 | 944.15 | 2026-02-26 15:15:00 | 896.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 09:15:00 | 944.90 | 2026-03-04 09:15:00 | 850.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 10:00:00 | 944.15 | 2026-03-04 09:15:00 | 849.74 | TARGET_HIT | 0.50 | 10.00% |
