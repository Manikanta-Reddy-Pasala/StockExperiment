# Hexaware Technologies Ltd. (HEXT)

## Backtest Summary

- **Window:** 2025-02-19 09:15:00 → 2026-05-08 15:15:00 (2081 bars)
- **Last close:** 486.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 13
- **Target hits / Stop hits / Partials:** 6 / 15 / 7
- **Avg / median % per leg:** 2.66% / 1.93%
- **Sum % (uncompounded):** 74.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.32% | -18.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.32% | -18.5% |
| SELL (all) | 20 | 15 | 75.0% | 6 | 7 | 7 | 4.65% | 93.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 15 | 75.0% | 6 | 7 | 7 | 4.65% | 93.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 15 | 53.6% | 6 | 15 | 7 | 2.66% | 74.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 797.05 | 735.96 | 735.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 801.85 | 737.78 | 736.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 843.25 | 843.70 | 814.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 843.25 | 843.70 | 814.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 827.75 | 849.15 | 825.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 827.75 | 849.15 | 825.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 826.80 | 848.93 | 825.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 779.50 | 848.93 | 825.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 773.00 | 848.17 | 825.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 763.55 | 848.17 | 825.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 697.30 | 806.16 | 806.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 694.00 | 805.05 | 806.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 762.30 | 759.52 | 778.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 762.75 | 759.52 | 778.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 774.95 | 759.75 | 777.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 796.85 | 759.75 | 777.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 796.95 | 760.12 | 777.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 789.85 | 771.47 | 781.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 788.15 | 771.88 | 781.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:00:00 | 787.55 | 772.03 | 781.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 750.36 | 771.68 | 781.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 748.74 | 771.68 | 781.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 748.17 | 771.68 | 781.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-05 12:15:00 | 710.87 | 764.46 | 775.88 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 772.90 | 759.83 | 772.45 | SL hit (close>ema200) qty=0.50 sl=759.83 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 772.90 | 759.83 | 772.45 | SL hit (close>ema200) qty=0.50 sl=759.83 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 747.85 | 763.17 | 771.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:15:00 | 710.46 | 760.36 | 769.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-25 15:15:00 | 673.07 | 748.86 | 762.61 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 741.00 | 718.92 | 738.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 740.75 | 718.92 | 738.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 734.45 | 719.08 | 738.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 732.40 | 719.08 | 738.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 743.00 | 719.69 | 738.80 | SL hit (close>static) qty=1.00 sl=742.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:00:00 | 731.40 | 720.00 | 738.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 730.45 | 719.73 | 736.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:30:00 | 732.30 | 721.32 | 735.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 12:15:00 | 694.83 | 718.60 | 732.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 12:15:00 | 693.93 | 718.60 | 732.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 12:15:00 | 695.68 | 718.60 | 732.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-10 09:15:00 | 658.26 | 710.28 | 726.14 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-10 09:15:00 | 657.41 | 710.28 | 726.14 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-10 09:15:00 | 659.07 | 710.28 | 726.14 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 722.00 | 702.89 | 718.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 722.00 | 702.89 | 718.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 721.15 | 703.07 | 718.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:00:00 | 719.50 | 704.65 | 718.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 723.70 | 704.84 | 718.67 | SL hit (close>static) qty=1.00 sl=723.65 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 715.65 | 705.71 | 718.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 12:45:00 | 718.95 | 706.11 | 718.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:30:00 | 719.80 | 706.54 | 718.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 727.05 | 706.75 | 718.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 727.05 | 706.75 | 718.79 | SL hit (close>static) qty=1.00 sl=723.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 727.05 | 706.75 | 718.79 | SL hit (close>static) qty=1.00 sl=723.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 727.05 | 706.75 | 718.79 | SL hit (close>static) qty=1.00 sl=723.65 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 727.05 | 706.75 | 718.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 727.85 | 707.94 | 719.04 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 11:15:00 | 765.85 | 726.81 | 726.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 773.00 | 735.45 | 731.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 742.20 | 744.45 | 737.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 15:15:00 | 742.20 | 744.45 | 737.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 742.20 | 744.45 | 737.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 738.40 | 744.45 | 737.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 736.60 | 744.37 | 737.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:15:00 | 733.45 | 744.37 | 737.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 734.00 | 744.27 | 737.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 733.40 | 744.27 | 737.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 736.60 | 744.03 | 737.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 739.20 | 744.03 | 737.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 738.00 | 743.97 | 737.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:30:00 | 737.00 | 743.97 | 737.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 740.65 | 743.89 | 737.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 737.00 | 743.89 | 737.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 738.55 | 744.11 | 737.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 739.30 | 744.11 | 737.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 742.15 | 744.09 | 737.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 749.40 | 744.06 | 737.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 14:15:00 | 750.40 | 744.11 | 737.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 751.65 | 744.29 | 737.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:15:00 | 748.00 | 744.10 | 738.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 738.45 | 746.56 | 740.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 738.45 | 746.56 | 740.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 739.80 | 746.49 | 740.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:15:00 | 737.80 | 746.49 | 740.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 737.45 | 746.40 | 740.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 736.95 | 746.31 | 740.12 | SL hit (close<static) qty=1.00 sl=737.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 736.95 | 746.31 | 740.12 | SL hit (close<static) qty=1.00 sl=737.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 736.95 | 746.31 | 740.12 | SL hit (close<static) qty=1.00 sl=737.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 736.95 | 746.31 | 740.12 | SL hit (close<static) qty=1.00 sl=737.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 743.40 | 746.06 | 740.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 735.90 | 745.96 | 740.06 | SL hit (close<static) qty=1.00 sl=736.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:30:00 | 742.50 | 745.81 | 740.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 13:15:00 | 735.55 | 745.70 | 740.02 | SL hit (close<static) qty=1.00 sl=736.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:30:00 | 742.40 | 745.45 | 740.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 742.90 | 745.40 | 740.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 742.20 | 745.52 | 740.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 742.20 | 745.52 | 740.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 737.60 | 745.45 | 740.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 737.20 | 745.45 | 740.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 737.35 | 745.36 | 740.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:30:00 | 737.95 | 745.36 | 740.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 706.60 | 744.90 | 740.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 706.60 | 744.90 | 740.07 | SL hit (close<static) qty=1.00 sl=736.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 706.60 | 744.90 | 740.07 | SL hit (close<static) qty=1.00 sl=736.05 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 707.35 | 744.90 | 740.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 738.05 | 742.31 | 739.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 738.05 | 742.31 | 739.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 728.70 | 742.17 | 739.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 728.70 | 742.17 | 739.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 730.45 | 742.06 | 739.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:00:00 | 730.45 | 742.06 | 739.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 736.50 | 739.65 | 737.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 738.35 | 739.65 | 737.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 738.95 | 739.64 | 737.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 738.95 | 739.64 | 737.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 739.55 | 739.64 | 737.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 728.35 | 739.64 | 737.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 729.60 | 739.54 | 737.92 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 708.00 | 736.21 | 736.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 691.20 | 730.13 | 733.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 722.85 | 721.12 | 727.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 12:15:00 | 727.30 | 721.19 | 727.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 727.30 | 721.19 | 727.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 13:00:00 | 727.30 | 721.19 | 727.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 13:15:00 | 728.30 | 721.26 | 727.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:00:00 | 728.30 | 721.26 | 727.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 727.85 | 721.32 | 727.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:45:00 | 731.35 | 721.32 | 727.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 720.00 | 721.31 | 727.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 703.70 | 721.31 | 727.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-05 09:15:00 | 633.33 | 718.34 | 725.84 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-26 15:00:00 | 789.85 | 2025-09-01 09:15:00 | 750.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 10:15:00 | 788.15 | 2025-09-01 09:15:00 | 748.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 11:00:00 | 787.55 | 2025-09-01 09:15:00 | 748.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 15:00:00 | 789.85 | 2025-09-05 12:15:00 | 710.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-28 10:15:00 | 788.15 | 2025-09-10 09:15:00 | 772.90 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-08-28 11:00:00 | 787.55 | 2025-09-10 09:15:00 | 772.90 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-09-22 09:15:00 | 747.85 | 2025-09-23 09:15:00 | 710.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 747.85 | 2025-09-25 15:15:00 | 673.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-15 12:15:00 | 732.40 | 2025-10-15 14:15:00 | 743.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-16 10:00:00 | 731.40 | 2025-10-31 12:15:00 | 694.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 12:15:00 | 730.45 | 2025-10-31 12:15:00 | 693.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 09:30:00 | 732.30 | 2025-10-31 12:15:00 | 695.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:00:00 | 731.40 | 2025-11-10 09:15:00 | 658.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-23 12:15:00 | 730.45 | 2025-11-10 09:15:00 | 657.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-28 09:30:00 | 732.30 | 2025-11-10 09:15:00 | 659.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 11:00:00 | 719.50 | 2025-11-20 11:15:00 | 723.70 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-21 09:15:00 | 715.65 | 2025-11-24 10:15:00 | 727.05 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-11-21 12:45:00 | 718.95 | 2025-11-24 10:15:00 | 727.05 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-24 09:30:00 | 719.80 | 2025-11-24 10:15:00 | 727.05 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-26 12:15:00 | 749.40 | 2026-01-05 13:15:00 | 736.95 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-12-26 14:15:00 | 750.40 | 2026-01-05 13:15:00 | 736.95 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-12-29 09:30:00 | 751.65 | 2026-01-05 13:15:00 | 736.95 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-12-30 12:15:00 | 748.00 | 2026-01-05 13:15:00 | 736.95 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-06 09:45:00 | 743.40 | 2026-01-06 10:15:00 | 735.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-01-06 12:30:00 | 742.50 | 2026-01-06 13:15:00 | 735.55 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-01-07 11:30:00 | 742.40 | 2026-01-09 09:15:00 | 706.60 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2026-01-07 12:30:00 | 742.90 | 2026-01-09 09:15:00 | 706.60 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-02-04 09:15:00 | 703.70 | 2026-02-05 09:15:00 | 633.33 | TARGET_HIT | 1.00 | 10.00% |
