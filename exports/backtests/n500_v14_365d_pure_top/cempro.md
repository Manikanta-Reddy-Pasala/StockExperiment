# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 955.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 0 |
| TARGET_HIT | 7 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 17
- **Target hits / Stop hits / Partials:** 7 / 17 / 0
- **Avg / median % per leg:** 1.30% / -1.34%
- **Sum % (uncompounded):** 31.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 7 | 30.4% | 7 | 16 | 0 | 1.46% | 33.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 7 | 30.4% | 7 | 16 | 0 | 1.46% | 33.7% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.51% | -2.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.51% | -2.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 7 | 29.2% | 7 | 17 | 0 | 1.30% | 31.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 631.00 | 534.45 | 533.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 634.00 | 536.33 | 534.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 831.40 | 832.68 | 763.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:45:00 | 833.20 | 832.68 | 763.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 772.75 | 823.70 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 773.15 | 823.70 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 771.70 | 823.18 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 771.95 | 823.18 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 772.00 | 822.67 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:30:00 | 772.90 | 822.67 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 769.15 | 822.14 | 772.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 769.15 | 822.14 | 772.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 770.50 | 821.63 | 772.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:15:00 | 766.50 | 821.63 | 772.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 766.50 | 821.08 | 771.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 777.10 | 821.08 | 771.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 779.10 | 820.66 | 772.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 786.45 | 810.28 | 770.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 788.00 | 810.28 | 770.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:30:00 | 785.00 | 809.56 | 771.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 806.50 | 808.44 | 771.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 776.75 | 806.90 | 773.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 771.55 | 806.90 | 773.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 774.00 | 806.58 | 773.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 778.20 | 806.58 | 773.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | SL hit (close<static) qty=1.00 sl=762.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | SL hit (close<static) qty=1.00 sl=762.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | SL hit (close<static) qty=1.00 sl=762.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 757.95 | 806.09 | 773.03 | SL hit (close<static) qty=1.00 sl=762.15 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 758.05 | 806.09 | 773.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 753.00 | 805.56 | 772.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 753.15 | 805.56 | 772.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 764.00 | 787.90 | 768.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 771.35 | 787.90 | 768.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:45:00 | 772.80 | 787.78 | 768.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:00:00 | 779.35 | 787.48 | 768.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 761.00 | 786.85 | 768.36 | SL hit (close<static) qty=1.00 sl=763.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 761.00 | 786.85 | 768.36 | SL hit (close<static) qty=1.00 sl=763.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 761.00 | 786.85 | 768.36 | SL hit (close<static) qty=1.00 sl=763.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 771.85 | 786.57 | 768.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 769.35 | 786.46 | 769.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 769.35 | 786.46 | 769.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 768.05 | 786.27 | 769.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 768.05 | 786.27 | 769.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 769.80 | 786.11 | 769.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 769.80 | 786.11 | 769.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 769.00 | 785.94 | 769.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 776.10 | 785.59 | 769.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:30:00 | 771.20 | 785.26 | 769.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 762.20 | 783.84 | 770.09 | SL hit (close<static) qty=1.00 sl=763.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 762.20 | 783.84 | 770.09 | SL hit (close<static) qty=1.00 sl=765.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 762.20 | 783.84 | 770.09 | SL hit (close<static) qty=1.00 sl=765.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 773.35 | 781.68 | 769.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 765.00 | 780.97 | 769.64 | SL hit (close<static) qty=1.00 sl=765.10 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 721.00 | 760.16 | 760.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 713.95 | 757.58 | 759.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 756.10 | 751.60 | 755.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 756.10 | 751.60 | 755.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 756.10 | 751.60 | 755.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 756.55 | 751.60 | 755.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 757.30 | 751.66 | 755.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 758.00 | 751.66 | 755.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 761.80 | 751.76 | 755.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 761.80 | 751.76 | 755.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 769.40 | 751.94 | 755.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 769.40 | 751.94 | 755.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 14:15:00 | 782.10 | 759.39 | 759.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 788.95 | 759.92 | 759.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 14:15:00 | 774.60 | 774.69 | 767.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 774.60 | 774.69 | 767.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 774.60 | 774.69 | 767.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 774.60 | 774.69 | 767.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 773.00 | 774.67 | 767.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 786.00 | 774.67 | 767.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:45:00 | 780.45 | 774.76 | 767.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 779.50 | 785.26 | 774.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-06 10:15:00 | 858.50 | 791.66 | 779.38 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-06 10:15:00 | 857.45 | 791.66 | 779.38 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 13:45:00 | 779.65 | 796.66 | 785.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 785.00 | 796.47 | 785.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 782.25 | 796.34 | 785.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 773.30 | 796.11 | 785.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 773.30 | 796.11 | 785.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 769.90 | 795.85 | 785.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 769.90 | 795.85 | 785.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 793.00 | 793.03 | 784.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 785.75 | 793.03 | 784.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-10-30 09:15:00 | 864.60 | 802.33 | 791.05 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-30 09:15:00 | 857.62 | 802.33 | 791.05 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 805.00 | 820.31 | 806.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 805.00 | 820.31 | 806.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 805.90 | 820.17 | 806.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 805.40 | 820.17 | 806.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 808.50 | 820.05 | 806.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:30:00 | 802.30 | 820.05 | 806.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 804.70 | 819.90 | 806.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 802.60 | 819.90 | 806.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 803.20 | 819.73 | 806.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 803.20 | 819.73 | 806.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 801.50 | 819.55 | 806.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:15:00 | 801.00 | 819.55 | 806.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 801.00 | 819.37 | 806.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 816.25 | 819.37 | 806.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 793.15 | 818.04 | 806.34 | SL hit (close<static) qty=1.00 sl=798.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 803.00 | 809.31 | 803.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:30:00 | 803.70 | 809.15 | 803.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 802.90 | 809.11 | 803.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-01 09:15:00 | 883.30 | 812.11 | 804.99 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-12-01 09:15:00 | 884.07 | 812.11 | 804.99 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-12-01 09:15:00 | 883.19 | 812.11 | 804.99 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 812.60 | 818.47 | 809.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 804.60 | 818.47 | 809.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 804.85 | 818.45 | 809.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 804.85 | 818.45 | 809.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 810.30 | 818.37 | 809.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 13:15:00 | 816.00 | 816.72 | 809.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 14:15:00 | 802.65 | 816.64 | 809.55 | SL hit (close<static) qty=1.00 sl=804.85 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 815.20 | 816.42 | 809.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:30:00 | 819.05 | 816.48 | 809.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:30:00 | 814.90 | 818.29 | 811.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 810.65 | 818.18 | 811.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 811.45 | 818.18 | 811.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 804.20 | 818.04 | 811.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 804.20 | 818.04 | 811.34 | SL hit (close<static) qty=1.00 sl=804.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 804.20 | 818.04 | 811.34 | SL hit (close<static) qty=1.00 sl=804.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 804.20 | 818.04 | 811.34 | SL hit (close<static) qty=1.00 sl=804.85 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 802.85 | 818.04 | 811.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 810.25 | 814.95 | 810.76 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 784.45 | 807.04 | 807.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 11:15:00 | 776.30 | 806.74 | 806.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 696.50 | 692.57 | 731.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 13:45:00 | 695.10 | 692.57 | 731.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 604.25 | 560.63 | 602.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:00:00 | 604.25 | 560.63 | 602.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 601.50 | 561.04 | 602.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 594.45 | 562.66 | 602.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 609.40 | 563.12 | 602.41 | SL hit (close>static) qty=1.00 sl=604.30 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 804.25 | 624.56 | 623.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 815.25 | 626.46 | 624.79 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-30 14:30:00 | 786.45 | 2025-08-05 09:15:00 | 757.95 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-07-30 15:15:00 | 788.00 | 2025-08-05 09:15:00 | 757.95 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2025-07-31 10:30:00 | 785.00 | 2025-08-05 09:15:00 | 757.95 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-08-01 09:15:00 | 806.50 | 2025-08-05 09:15:00 | 757.95 | STOP_HIT | 1.00 | -6.02% |
| BUY | retest2 | 2025-08-12 09:15:00 | 771.35 | 2025-08-12 14:15:00 | 761.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-08-12 09:45:00 | 772.80 | 2025-08-12 14:15:00 | 761.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-08-12 12:00:00 | 779.35 | 2025-08-12 14:15:00 | 761.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-08-13 09:15:00 | 771.85 | 2025-08-21 12:15:00 | 762.20 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-08-18 09:15:00 | 776.10 | 2025-08-21 12:15:00 | 762.20 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-08-18 12:30:00 | 771.20 | 2025-08-21 12:15:00 | 762.20 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-25 10:00:00 | 773.35 | 2025-08-25 14:15:00 | 765.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-22 09:15:00 | 786.00 | 2025-10-06 10:15:00 | 858.50 | TARGET_HIT | 1.00 | 9.22% |
| BUY | retest2 | 2025-09-22 09:45:00 | 780.45 | 2025-10-06 10:15:00 | 857.45 | TARGET_HIT | 1.00 | 9.87% |
| BUY | retest2 | 2025-09-29 15:00:00 | 779.50 | 2025-10-30 09:15:00 | 864.60 | TARGET_HIT | 1.00 | 10.92% |
| BUY | retest2 | 2025-10-15 13:45:00 | 779.65 | 2025-10-30 09:15:00 | 857.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-19 09:15:00 | 816.25 | 2025-11-21 10:15:00 | 793.15 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-11-27 11:00:00 | 803.00 | 2025-12-01 09:15:00 | 883.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-27 12:30:00 | 803.70 | 2025-12-01 09:15:00 | 884.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-27 13:45:00 | 802.90 | 2025-12-01 09:15:00 | 883.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-09 13:15:00 | 816.00 | 2025-12-10 14:15:00 | 802.65 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-12-11 10:15:00 | 815.20 | 2025-12-17 10:15:00 | 804.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-11 11:30:00 | 819.05 | 2025-12-17 10:15:00 | 804.20 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-12-16 13:30:00 | 814.90 | 2025-12-17 10:15:00 | 804.20 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-09 09:45:00 | 594.45 | 2026-04-09 10:15:00 | 609.40 | STOP_HIT | 1.00 | -2.51% |
