# Five-Star Business Finance Ltd. (FIVESTAR)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 462.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 16 |
| ALERT2 | 15 |
| ALERT2_SKIP | 6 |
| ALERT3 | 72 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 51
- **Target hits / Stop hits / Partials:** 2 / 54 / 0
- **Avg / median % per leg:** -2.46% / -2.24%
- **Sum % (uncompounded):** -137.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 4 | 14.3% | 1 | 27 | 0 | -2.33% | -65.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 4 | 14.3% | 1 | 27 | 0 | -2.33% | -65.3% |
| SELL (all) | 28 | 1 | 3.6% | 1 | 27 | 0 | -2.59% | -72.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 1 | 3.6% | 1 | 27 | 0 | -2.59% | -72.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 56 | 5 | 8.9% | 2 | 54 | 0 | -2.46% | -137.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 10:15:00 | 689.95 | 742.17 | 742.35 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 14:15:00 | 759.70 | 737.02 | 737.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 769.85 | 740.52 | 738.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 749.85 | 754.68 | 747.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 14:15:00 | 749.85 | 754.68 | 747.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 749.85 | 754.68 | 747.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 15:00:00 | 749.85 | 754.68 | 747.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 735.05 | 754.48 | 747.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:30:00 | 743.15 | 754.38 | 747.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 760.80 | 754.44 | 747.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 15:15:00 | 765.00 | 754.36 | 747.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 13:00:00 | 763.95 | 754.30 | 747.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 09:15:00 | 763.35 | 754.14 | 747.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 14:30:00 | 764.95 | 754.85 | 748.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 745.40 | 756.00 | 749.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:45:00 | 744.65 | 756.00 | 749.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 742.80 | 755.86 | 749.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:30:00 | 740.25 | 755.86 | 749.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-20 09:15:00 | 729.60 | 754.94 | 748.80 | SL hit (close<static) qty=1.00 sl=732.30 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 11:15:00 | 709.40 | 743.67 | 743.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 708.50 | 740.37 | 742.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 10:15:00 | 687.85 | 687.40 | 709.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-21 11:00:00 | 687.85 | 687.40 | 709.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 717.05 | 688.98 | 708.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 15:00:00 | 717.05 | 688.98 | 708.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 710.00 | 689.18 | 708.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 09:15:00 | 702.50 | 689.18 | 708.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 14:15:00 | 729.70 | 691.57 | 708.88 | SL hit (close>static) qty=1.00 sl=720.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 13:15:00 | 773.60 | 717.41 | 717.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 09:15:00 | 783.90 | 719.18 | 718.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 14:15:00 | 719.10 | 728.27 | 723.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 14:15:00 | 719.10 | 728.27 | 723.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 719.10 | 728.27 | 723.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 15:00:00 | 719.10 | 728.27 | 723.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 728.95 | 728.27 | 723.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:15:00 | 720.40 | 728.27 | 723.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 713.95 | 728.13 | 723.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:30:00 | 713.35 | 728.13 | 723.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 727.05 | 728.12 | 723.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:45:00 | 718.55 | 728.12 | 723.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 724.75 | 728.09 | 723.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:30:00 | 722.00 | 728.09 | 723.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 725.45 | 728.06 | 723.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 14:00:00 | 729.95 | 728.08 | 723.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 15:15:00 | 732.00 | 728.08 | 723.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 14:15:00 | 728.25 | 728.24 | 723.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 14:15:00 | 719.90 | 728.16 | 723.56 | SL hit (close<static) qty=1.00 sl=722.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 14:15:00 | 720.20 | 765.47 | 765.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 10:15:00 | 716.30 | 761.47 | 763.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 738.85 | 738.43 | 749.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-23 11:45:00 | 738.25 | 738.43 | 749.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 743.45 | 738.43 | 748.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:30:00 | 740.15 | 738.41 | 747.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 15:15:00 | 736.65 | 738.35 | 747.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:45:00 | 739.55 | 738.37 | 747.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:30:00 | 739.15 | 738.39 | 747.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 753.75 | 738.54 | 747.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-30 12:15:00 | 753.75 | 738.54 | 747.59 | SL hit (close>static) qty=1.00 sl=748.95 alert=retest2 |

### Cycle 6 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 799.65 | 752.41 | 752.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 09:15:00 | 812.95 | 758.16 | 755.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 781.10 | 833.80 | 806.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 781.10 | 833.80 | 806.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 781.10 | 833.80 | 806.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 781.10 | 833.80 | 806.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 771.55 | 833.18 | 806.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 771.55 | 833.18 | 806.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 652.50 | 784.18 | 784.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 648.75 | 782.83 | 783.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 670.00 | 667.86 | 702.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:30:00 | 667.85 | 667.86 | 702.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 700.30 | 668.80 | 702.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:45:00 | 699.30 | 668.80 | 702.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 710.00 | 669.21 | 702.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 710.60 | 669.21 | 702.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 715.25 | 669.67 | 702.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:45:00 | 718.65 | 669.67 | 702.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 729.05 | 670.26 | 702.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 729.05 | 670.26 | 702.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 802.00 | 724.34 | 724.19 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 652.75 | 725.29 | 725.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 644.65 | 723.89 | 724.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 14:15:00 | 699.65 | 696.53 | 708.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:45:00 | 697.90 | 696.53 | 708.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 702.00 | 696.58 | 708.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 672.75 | 696.58 | 708.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 11:15:00 | 696.20 | 696.65 | 708.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 10:15:00 | 720.40 | 697.38 | 708.44 | SL hit (close>static) qty=1.00 sl=715.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 11:15:00 | 756.00 | 717.07 | 716.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 14:15:00 | 762.55 | 718.35 | 717.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 13:15:00 | 721.45 | 722.65 | 719.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 14:00:00 | 721.45 | 722.65 | 719.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 728.25 | 722.70 | 719.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:45:00 | 739.00 | 722.70 | 719.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 695.45 | 722.47 | 719.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:45:00 | 695.75 | 722.47 | 719.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 694.05 | 722.19 | 719.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 11:00:00 | 694.05 | 722.19 | 719.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 714.55 | 718.57 | 717.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:30:00 | 712.70 | 718.57 | 717.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 708.45 | 718.47 | 717.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 708.45 | 718.47 | 717.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 700.00 | 718.27 | 717.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:00:00 | 700.00 | 718.27 | 717.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 697.25 | 718.06 | 717.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:30:00 | 701.90 | 718.06 | 717.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 685.00 | 717.36 | 717.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 667.00 | 708.33 | 711.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 704.00 | 693.47 | 702.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 10:15:00 | 704.00 | 693.47 | 702.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 704.00 | 693.47 | 702.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:00:00 | 704.00 | 693.47 | 702.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 710.25 | 693.64 | 702.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:00:00 | 710.25 | 693.64 | 702.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 712.50 | 694.91 | 703.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 702.00 | 698.67 | 704.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:00:00 | 706.75 | 698.75 | 704.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 728.95 | 699.87 | 704.51 | SL hit (close>static) qty=1.00 sl=716.95 alert=retest2 |

### Cycle 12 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 10:15:00 | 754.15 | 706.27 | 706.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 12:15:00 | 759.40 | 707.25 | 706.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 731.10 | 743.69 | 727.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 731.10 | 743.69 | 727.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 731.10 | 743.69 | 727.87 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 677.40 | 716.86 | 716.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 10:15:00 | 675.50 | 715.03 | 715.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 703.60 | 701.97 | 708.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 09:30:00 | 705.40 | 701.97 | 708.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 708.75 | 702.06 | 708.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 709.35 | 702.06 | 708.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 707.70 | 702.11 | 708.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 707.00 | 702.11 | 708.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 706.65 | 702.16 | 708.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 701.20 | 702.26 | 708.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 713.40 | 702.37 | 708.33 | SL hit (close>static) qty=1.00 sl=709.45 alert=retest2 |

### Cycle 14 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 798.10 | 710.94 | 710.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 800.05 | 711.83 | 711.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 723.85 | 733.71 | 724.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 723.85 | 733.71 | 724.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 723.85 | 733.71 | 724.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 723.85 | 733.71 | 724.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 715.00 | 733.52 | 724.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 735.90 | 733.52 | 724.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:30:00 | 732.30 | 733.19 | 724.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:00:00 | 724.50 | 750.27 | 740.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:45:00 | 724.85 | 748.66 | 740.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 699.55 | 743.91 | 738.41 | SL hit (close<static) qty=1.00 sl=709.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 650.05 | 733.35 | 733.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 13:15:00 | 645.15 | 732.47 | 732.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 540.30 | 536.65 | 567.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 12:45:00 | 540.60 | 536.65 | 567.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 584.10 | 536.46 | 561.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 584.10 | 536.46 | 561.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 602.10 | 537.12 | 561.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 600.60 | 537.12 | 561.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 625.95 | 581.08 | 580.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 628.45 | 584.22 | 582.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 596.80 | 597.25 | 590.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 12:30:00 | 598.50 | 597.25 | 590.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 591.25 | 597.12 | 590.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 584.65 | 597.12 | 590.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 591.75 | 600.17 | 592.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 591.75 | 600.17 | 592.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 588.20 | 600.05 | 592.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 588.20 | 600.05 | 592.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 590.00 | 599.95 | 592.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:30:00 | 595.10 | 599.88 | 592.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 592.00 | 599.78 | 592.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 592.40 | 599.78 | 592.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:30:00 | 594.10 | 599.61 | 592.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 589.45 | 599.51 | 592.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 585.50 | 599.37 | 592.83 | SL hit (close<static) qty=1.00 sl=587.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 567.70 | 588.18 | 588.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 553.50 | 584.68 | 586.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 404.20 | 386.12 | 423.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 09:45:00 | 405.40 | 386.12 | 423.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 424.00 | 389.09 | 422.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 424.00 | 389.09 | 422.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 422.55 | 389.42 | 422.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 427.85 | 389.42 | 422.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 428.85 | 389.81 | 422.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 412.60 | 392.18 | 422.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 421.10 | 392.67 | 422.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 421.90 | 392.96 | 422.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:15:00 | 420.90 | 393.53 | 422.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 420.65 | 393.80 | 422.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 437.80 | 394.92 | 422.90 | SL hit (close>static) qty=1.00 sl=434.50 alert=retest2 |

### Cycle 18 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 495.00 | 442.04 | 441.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 15:15:00 | 502.00 | 444.70 | 443.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-02-13 15:15:00 | 765.00 | 2024-02-20 09:15:00 | 729.60 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2024-02-14 13:00:00 | 763.95 | 2024-02-20 09:15:00 | 729.60 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2024-02-15 09:15:00 | 763.35 | 2024-02-20 09:15:00 | 729.60 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2024-02-15 14:30:00 | 764.95 | 2024-02-20 09:15:00 | 729.60 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2024-03-26 09:15:00 | 702.50 | 2024-03-27 14:15:00 | 729.70 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2024-04-12 09:30:00 | 707.60 | 2024-04-12 13:15:00 | 721.85 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-04-12 11:30:00 | 709.90 | 2024-04-12 13:15:00 | 721.85 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-04-12 12:00:00 | 709.95 | 2024-04-12 13:15:00 | 721.85 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-04-26 13:00:00 | 709.25 | 2024-04-26 14:15:00 | 734.10 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-05-10 14:00:00 | 729.95 | 2024-05-13 14:15:00 | 719.90 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-05-10 15:15:00 | 732.00 | 2024-05-13 14:15:00 | 719.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-05-13 14:15:00 | 728.25 | 2024-05-13 14:15:00 | 719.90 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-05-14 11:00:00 | 729.20 | 2024-05-22 12:15:00 | 721.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-06-06 09:15:00 | 767.35 | 2024-06-07 09:15:00 | 844.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-19 11:15:00 | 759.25 | 2024-07-30 10:15:00 | 763.00 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2024-07-23 09:45:00 | 757.15 | 2024-08-02 09:15:00 | 759.50 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-07-23 12:45:00 | 758.10 | 2024-08-02 09:15:00 | 759.50 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-07-29 14:45:00 | 771.00 | 2024-08-05 09:15:00 | 752.40 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-08-01 10:00:00 | 770.70 | 2024-08-07 14:15:00 | 720.20 | STOP_HIT | 1.00 | -6.55% |
| BUY | retest2 | 2024-08-01 11:30:00 | 771.00 | 2024-08-07 14:15:00 | 720.20 | STOP_HIT | 1.00 | -6.59% |
| BUY | retest2 | 2024-08-02 12:15:00 | 778.10 | 2024-08-07 14:15:00 | 720.20 | STOP_HIT | 1.00 | -7.44% |
| SELL | retest2 | 2024-08-29 10:30:00 | 740.15 | 2024-08-30 12:15:00 | 753.75 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-08-29 15:15:00 | 736.65 | 2024-08-30 12:15:00 | 753.75 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-08-30 10:45:00 | 739.55 | 2024-08-30 12:15:00 | 753.75 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-08-30 11:30:00 | 739.15 | 2024-08-30 12:15:00 | 753.75 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-08-30 14:45:00 | 750.80 | 2024-08-30 15:15:00 | 761.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-09-06 09:45:00 | 753.20 | 2024-09-16 09:15:00 | 761.10 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-09-06 13:45:00 | 752.60 | 2024-09-16 09:15:00 | 761.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-01-27 09:15:00 | 672.75 | 2025-01-28 10:15:00 | 720.40 | STOP_HIT | 1.00 | -7.08% |
| SELL | retest2 | 2025-01-27 11:15:00 | 696.20 | 2025-01-28 10:15:00 | 720.40 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-03-25 09:15:00 | 702.00 | 2025-03-27 15:15:00 | 728.95 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-03-25 10:00:00 | 706.75 | 2025-03-27 15:15:00 | 728.95 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-04-01 09:15:00 | 703.75 | 2025-04-07 09:15:00 | 633.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 15:15:00 | 709.00 | 2025-04-11 14:15:00 | 717.25 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-04-11 10:15:00 | 691.65 | 2025-04-15 09:15:00 | 734.65 | STOP_HIT | 1.00 | -6.22% |
| SELL | retest2 | 2025-05-27 09:15:00 | 701.20 | 2025-05-27 09:15:00 | 713.40 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-05-28 09:30:00 | 705.00 | 2025-06-02 09:15:00 | 720.80 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-05-28 10:00:00 | 704.85 | 2025-06-02 09:15:00 | 720.80 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-05-28 11:30:00 | 702.10 | 2025-06-02 09:15:00 | 720.80 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-06-02 15:00:00 | 701.00 | 2025-06-06 12:15:00 | 746.15 | STOP_HIT | 1.00 | -6.44% |
| BUY | retest2 | 2025-06-23 09:15:00 | 735.90 | 2025-07-25 10:15:00 | 699.55 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest2 | 2025-06-23 14:30:00 | 732.30 | 2025-07-25 10:15:00 | 699.55 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-07-21 11:00:00 | 724.50 | 2025-07-25 10:15:00 | 699.55 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-07-22 11:45:00 | 724.85 | 2025-07-25 10:15:00 | 699.55 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-11-27 13:30:00 | 595.10 | 2025-12-01 09:15:00 | 585.50 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-11-27 15:15:00 | 592.00 | 2025-12-01 09:15:00 | 585.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-28 13:00:00 | 592.40 | 2025-12-01 09:15:00 | 585.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-28 14:30:00 | 594.10 | 2025-12-01 09:15:00 | 585.50 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-04 10:30:00 | 592.40 | 2025-12-05 09:15:00 | 580.55 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-12-04 13:15:00 | 595.10 | 2025-12-05 09:15:00 | 580.55 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-12-04 14:00:00 | 592.50 | 2025-12-05 09:15:00 | 580.55 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-12-04 15:00:00 | 591.20 | 2025-12-05 09:15:00 | 580.55 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-04-13 09:15:00 | 412.60 | 2026-04-15 10:15:00 | 437.80 | STOP_HIT | 1.00 | -6.11% |
| SELL | retest2 | 2026-04-13 11:00:00 | 421.10 | 2026-04-15 10:15:00 | 437.80 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-04-13 12:15:00 | 421.90 | 2026-04-15 10:15:00 | 437.80 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2026-04-13 14:15:00 | 420.90 | 2026-04-15 10:15:00 | 437.80 | STOP_HIT | 1.00 | -4.02% |
