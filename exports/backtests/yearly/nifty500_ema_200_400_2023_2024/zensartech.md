# Zensar Technolgies Ltd. (ZENSARTECH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 525.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 11 |
| TARGET_HIT | 12 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 21
- **Target hits / Stop hits / Partials:** 12 / 32 / 11
- **Avg / median % per leg:** 2.27% / 1.64%
- **Sum % (uncompounded):** 125.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 12 | 52.2% | 12 | 11 | 0 | 3.70% | 85.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 12 | 52.2% | 12 | 11 | 0 | 3.70% | 85.0% |
| SELL (all) | 32 | 22 | 68.8% | 0 | 21 | 11 | 1.25% | 40.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 22 | 68.8% | 0 | 21 | 11 | 1.25% | 40.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 34 | 61.8% | 12 | 32 | 11 | 2.27% | 125.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 530.00 | 555.16 | 555.29 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 15:15:00 | 562.60 | 553.76 | 553.76 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 09:15:00 | 544.50 | 553.67 | 553.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 532.65 | 553.36 | 553.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 14:15:00 | 552.80 | 551.98 | 552.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 14:15:00 | 552.80 | 551.98 | 552.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 552.80 | 551.98 | 552.83 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 15:15:00 | 585.95 | 553.75 | 553.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 590.55 | 556.80 | 555.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 14:15:00 | 590.20 | 591.53 | 577.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-16 15:00:00 | 590.20 | 591.53 | 577.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 578.90 | 591.40 | 578.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 15:00:00 | 578.90 | 591.40 | 578.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 584.25 | 591.32 | 578.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 572.90 | 591.32 | 578.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 578.95 | 591.20 | 578.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 12:45:00 | 581.25 | 590.80 | 578.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 14:15:00 | 580.00 | 590.69 | 578.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 587.40 | 590.41 | 578.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 14:45:00 | 580.00 | 587.71 | 577.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 582.45 | 587.60 | 578.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:45:00 | 577.05 | 587.60 | 578.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 577.80 | 587.51 | 578.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 580.05 | 587.51 | 578.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 578.95 | 587.42 | 578.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:30:00 | 577.05 | 587.42 | 578.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 576.60 | 587.31 | 578.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:45:00 | 577.00 | 587.31 | 578.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 579.40 | 587.24 | 578.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 597.70 | 587.00 | 578.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-26 09:15:00 | 638.00 | 587.50 | 578.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 674.95 | 744.32 | 744.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 12:15:00 | 669.95 | 743.58 | 744.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 723.35 | 715.38 | 727.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 15:00:00 | 723.35 | 715.38 | 727.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 699.05 | 701.28 | 714.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:30:00 | 713.60 | 701.28 | 714.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 712.55 | 701.42 | 714.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:00:00 | 712.55 | 701.42 | 714.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 716.40 | 701.57 | 714.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 13:00:00 | 716.40 | 701.57 | 714.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 716.80 | 701.72 | 714.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 13:45:00 | 717.95 | 701.72 | 714.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 712.15 | 701.82 | 714.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 712.15 | 701.82 | 714.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 720.45 | 702.06 | 713.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 720.45 | 702.06 | 713.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 728.10 | 702.32 | 713.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 728.10 | 702.32 | 713.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 725.00 | 702.54 | 713.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 720.45 | 708.21 | 715.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 729.30 | 708.60 | 715.92 | SL hit (close>static) qty=1.00 sl=729.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 12:15:00 | 754.25 | 719.95 | 719.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 759.15 | 721.38 | 720.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 765.20 | 766.53 | 748.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 765.20 | 766.53 | 748.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 765.20 | 766.58 | 749.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:45:00 | 773.95 | 759.23 | 748.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 14:45:00 | 776.30 | 763.72 | 752.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 789.90 | 763.81 | 752.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 15:00:00 | 774.20 | 764.49 | 752.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 13:15:00 | 750.50 | 766.89 | 755.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 13:45:00 | 746.65 | 766.89 | 755.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 748.75 | 766.71 | 755.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:30:00 | 750.70 | 766.71 | 755.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 15:15:00 | 738.85 | 766.44 | 755.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-13 15:15:00 | 738.85 | 766.44 | 755.52 | SL hit (close<static) qty=1.00 sl=746.15 alert=retest2 |

### Cycle 7 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 745.90 | 793.72 | 793.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 738.05 | 791.68 | 792.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 14:15:00 | 688.90 | 687.89 | 719.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-21 15:00:00 | 688.90 | 687.89 | 719.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 713.20 | 689.59 | 716.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 15:15:00 | 694.55 | 700.17 | 717.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 11:45:00 | 698.20 | 700.03 | 716.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 719.00 | 700.11 | 716.55 | SL hit (close>static) qty=1.00 sl=718.10 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 783.80 | 727.87 | 727.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 861.95 | 730.37 | 729.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 832.00 | 839.88 | 815.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 832.00 | 839.88 | 815.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 817.40 | 839.08 | 816.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 819.95 | 839.08 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 816.00 | 838.85 | 816.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 811.00 | 838.85 | 816.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 814.20 | 838.60 | 816.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 823.75 | 835.13 | 815.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:45:00 | 821.05 | 834.40 | 816.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 795.45 | 833.22 | 817.89 | SL hit (close<static) qty=1.00 sl=809.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 800.00 | 809.62 | 809.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 796.20 | 808.65 | 809.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 814.65 | 807.65 | 808.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 11:15:00 | 814.65 | 807.65 | 808.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 814.65 | 807.65 | 808.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 814.65 | 807.65 | 808.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 815.75 | 807.73 | 808.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:30:00 | 810.50 | 807.73 | 808.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 814.95 | 808.05 | 808.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 814.50 | 808.05 | 808.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 810.85 | 808.22 | 808.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 807.00 | 808.22 | 808.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 802.00 | 807.85 | 808.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:45:00 | 808.05 | 807.86 | 808.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 807.10 | 807.80 | 808.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 13:15:00 | 767.65 | 804.62 | 806.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 13:15:00 | 766.75 | 804.62 | 806.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 15:15:00 | 766.65 | 803.85 | 806.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 15:15:00 | 761.90 | 803.85 | 806.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 801.95 | 794.28 | 800.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 801.95 | 794.28 | 800.56 | SL hit (close>ema200) qty=0.50 sl=794.28 alert=retest2 |

### Cycle 10 — BUY (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 12:15:00 | 853.95 | 805.34 | 805.33 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 759.20 | 806.04 | 806.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 13:15:00 | 757.00 | 805.55 | 805.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 793.30 | 789.54 | 796.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:30:00 | 791.45 | 789.54 | 796.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 792.15 | 789.57 | 796.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 790.20 | 789.57 | 796.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 790.50 | 789.59 | 796.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 800.95 | 789.76 | 796.59 | SL hit (close>static) qty=1.00 sl=797.55 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-19 12:45:00 | 581.25 | 2024-04-26 09:15:00 | 638.00 | TARGET_HIT | 1.00 | 9.76% |
| BUY | retest2 | 2024-04-19 14:15:00 | 580.00 | 2024-04-26 09:15:00 | 638.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-22 09:15:00 | 587.40 | 2024-04-26 10:15:00 | 639.38 | TARGET_HIT | 1.00 | 8.85% |
| BUY | retest2 | 2024-04-24 14:45:00 | 580.00 | 2024-05-03 09:15:00 | 646.14 | TARGET_HIT | 1.00 | 11.40% |
| BUY | retest2 | 2024-04-26 09:15:00 | 597.70 | 2024-05-03 09:15:00 | 657.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-09 15:15:00 | 582.00 | 2024-05-16 09:15:00 | 640.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-10 09:45:00 | 583.95 | 2024-05-16 09:15:00 | 642.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 13:30:00 | 582.85 | 2024-06-06 13:15:00 | 641.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 11:45:00 | 612.00 | 2024-06-06 15:15:00 | 673.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 14:15:00 | 610.90 | 2024-06-06 15:15:00 | 671.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 09:15:00 | 621.00 | 2024-06-07 09:15:00 | 683.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-11 09:15:00 | 720.45 | 2024-11-11 10:15:00 | 729.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-11-13 09:15:00 | 715.55 | 2024-11-25 10:15:00 | 731.95 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-11-22 12:00:00 | 722.50 | 2024-11-25 10:15:00 | 731.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-11-22 15:15:00 | 722.00 | 2024-11-25 10:15:00 | 731.95 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-01-02 11:45:00 | 773.95 | 2025-01-13 15:15:00 | 738.85 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest2 | 2025-01-06 14:45:00 | 776.30 | 2025-01-13 15:15:00 | 738.85 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2025-01-07 09:15:00 | 789.90 | 2025-01-13 15:15:00 | 738.85 | STOP_HIT | 1.00 | -6.46% |
| BUY | retest2 | 2025-01-07 15:00:00 | 774.20 | 2025-01-13 15:15:00 | 738.85 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest2 | 2025-01-20 12:15:00 | 753.70 | 2025-01-21 09:15:00 | 747.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-20 14:45:00 | 754.60 | 2025-01-21 09:15:00 | 747.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-01-22 15:15:00 | 755.95 | 2025-01-23 09:15:00 | 831.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-24 10:30:00 | 757.80 | 2025-02-28 10:15:00 | 744.30 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-06 15:15:00 | 694.55 | 2025-05-08 09:15:00 | 719.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-05-07 11:45:00 | 698.20 | 2025-05-08 09:15:00 | 719.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-05-09 09:15:00 | 691.60 | 2025-05-12 09:15:00 | 736.00 | STOP_HIT | 1.00 | -6.42% |
| SELL | retest2 | 2025-05-09 11:45:00 | 698.00 | 2025-05-12 09:15:00 | 736.00 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2025-07-16 09:15:00 | 823.75 | 2025-07-23 09:15:00 | 795.45 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-07-18 10:45:00 | 821.05 | 2025-07-23 09:15:00 | 795.45 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-07-24 10:00:00 | 825.20 | 2025-07-25 10:15:00 | 802.10 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-08-12 10:00:00 | 820.00 | 2025-08-12 13:15:00 | 806.20 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-08-21 15:15:00 | 807.00 | 2025-08-29 13:15:00 | 767.65 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2025-08-26 09:15:00 | 802.00 | 2025-08-29 13:15:00 | 766.75 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-08-26 11:45:00 | 808.05 | 2025-08-29 15:15:00 | 766.65 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2025-08-26 15:00:00 | 807.10 | 2025-08-29 15:15:00 | 761.90 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-08-21 15:15:00 | 807.00 | 2025-09-10 09:15:00 | 801.95 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2025-08-26 09:15:00 | 802.00 | 2025-09-10 09:15:00 | 801.95 | STOP_HIT | 0.50 | 0.01% |
| SELL | retest2 | 2025-08-26 11:45:00 | 808.05 | 2025-09-10 09:15:00 | 801.95 | STOP_HIT | 0.50 | 0.75% |
| SELL | retest2 | 2025-08-26 15:00:00 | 807.10 | 2025-09-10 09:15:00 | 801.95 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2025-10-10 11:15:00 | 790.20 | 2025-10-10 14:15:00 | 800.95 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-10 12:15:00 | 790.50 | 2025-10-10 14:15:00 | 800.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-10-13 09:15:00 | 789.40 | 2025-10-15 11:15:00 | 749.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 789.40 | 2025-10-17 11:15:00 | 784.05 | STOP_HIT | 0.50 | 0.68% |
| SELL | retest2 | 2025-11-03 09:15:00 | 757.15 | 2025-11-06 11:15:00 | 719.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 09:15:00 | 757.15 | 2025-11-27 11:15:00 | 743.80 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-12-05 12:00:00 | 747.25 | 2025-12-11 09:15:00 | 709.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 12:00:00 | 747.25 | 2025-12-19 10:15:00 | 740.25 | STOP_HIT | 0.50 | 0.94% |
| SELL | retest2 | 2025-12-05 13:30:00 | 745.60 | 2025-12-30 09:15:00 | 708.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 09:45:00 | 746.00 | 2025-12-30 09:15:00 | 708.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 10:30:00 | 747.25 | 2025-12-30 09:15:00 | 709.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 15:00:00 | 745.35 | 2025-12-30 09:15:00 | 708.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 13:30:00 | 745.60 | 2026-01-09 10:15:00 | 733.15 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2025-12-22 09:45:00 | 746.00 | 2026-01-09 10:15:00 | 733.15 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-12-22 10:30:00 | 747.25 | 2026-01-09 10:15:00 | 733.15 | STOP_HIT | 0.50 | 1.89% |
| SELL | retest2 | 2025-12-22 15:00:00 | 745.35 | 2026-01-09 10:15:00 | 733.15 | STOP_HIT | 0.50 | 1.64% |
