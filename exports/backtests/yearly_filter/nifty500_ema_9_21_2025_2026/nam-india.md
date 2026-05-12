# Nippon Life India Asset Management Ltd. (NAM-INDIA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1100.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 53 |
| ALERT2 | 54 |
| ALERT2_SKIP | 32 |
| ALERT3 | 129 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 44 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 31
- **Target hits / Stop hits / Partials:** 2 / 48 / 3
- **Avg / median % per leg:** 0.08% / -1.10%
- **Sum % (uncompounded):** 3.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 19 | 61.3% | 2 | 26 | 3 | 1.33% | 41.3% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.67% | 22.0% |
| BUY @ 3rd Alert (retest2) | 25 | 13 | 52.0% | 2 | 23 | 0 | 0.77% | 19.3% |
| SELL (all) | 22 | 3 | 13.6% | 0 | 22 | 0 | -1.70% | -37.3% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | -1.22% | -3.7% |
| SELL @ 3rd Alert (retest2) | 19 | 1 | 5.3% | 0 | 19 | 0 | -1.77% | -33.7% |
| retest1 (combined) | 9 | 8 | 88.9% | 0 | 6 | 3 | 2.04% | 18.3% |
| retest2 (combined) | 44 | 14 | 31.8% | 2 | 42 | 0 | -0.33% | -14.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 669.55 | 644.23 | 642.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 688.00 | 652.98 | 646.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 695.30 | 696.08 | 681.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:45:00 | 696.00 | 696.08 | 681.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 744.00 | 740.50 | 728.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 729.15 | 740.50 | 728.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 743.65 | 743.48 | 737.78 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 726.50 | 735.51 | 736.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 721.80 | 732.77 | 734.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 724.85 | 724.48 | 729.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 724.85 | 724.48 | 729.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 730.20 | 725.62 | 729.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 730.95 | 725.62 | 729.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 723.85 | 725.27 | 728.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:15:00 | 720.10 | 725.27 | 728.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 721.00 | 726.67 | 727.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 12:15:00 | 736.00 | 729.49 | 728.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 736.00 | 729.49 | 728.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 741.00 | 732.67 | 730.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 733.00 | 737.94 | 734.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 733.00 | 737.94 | 734.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 733.00 | 737.94 | 734.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 733.00 | 737.94 | 734.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 735.60 | 737.47 | 734.65 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 732.95 | 733.86 | 733.90 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 742.10 | 735.33 | 734.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 754.45 | 742.53 | 739.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 738.45 | 745.26 | 741.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 738.45 | 745.26 | 741.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 738.45 | 745.26 | 741.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 738.45 | 745.26 | 741.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 735.00 | 743.20 | 741.18 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 730.00 | 739.17 | 739.61 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 746.20 | 739.57 | 739.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 765.15 | 748.44 | 743.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 788.95 | 798.39 | 784.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:30:00 | 793.50 | 798.39 | 784.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 779.95 | 794.70 | 784.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 779.95 | 794.70 | 784.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 780.90 | 791.94 | 784.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 781.55 | 791.94 | 784.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 762.85 | 780.24 | 780.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 756.60 | 772.63 | 776.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 735.15 | 732.05 | 743.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 735.15 | 732.05 | 743.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 735.65 | 732.32 | 739.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 735.65 | 732.32 | 739.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 740.60 | 733.98 | 739.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 740.60 | 733.98 | 739.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 745.45 | 736.27 | 740.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 745.45 | 736.27 | 740.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 745.00 | 738.02 | 740.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 743.05 | 738.02 | 740.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 750.85 | 742.74 | 742.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 13:15:00 | 763.05 | 750.94 | 746.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 746.25 | 753.65 | 749.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 746.25 | 753.65 | 749.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 746.25 | 753.65 | 749.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 746.25 | 753.65 | 749.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 748.70 | 752.66 | 749.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:15:00 | 742.80 | 752.66 | 749.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 745.60 | 751.25 | 749.24 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 742.20 | 746.88 | 747.48 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 753.80 | 748.26 | 748.06 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 740.05 | 746.79 | 747.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 14:15:00 | 734.85 | 743.30 | 745.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 752.10 | 743.40 | 745.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 752.10 | 743.40 | 745.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 752.10 | 743.40 | 745.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 752.10 | 743.40 | 745.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 757.05 | 746.13 | 746.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 757.05 | 746.13 | 746.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 756.80 | 748.26 | 747.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 12:15:00 | 761.30 | 750.87 | 748.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 781.05 | 787.60 | 781.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 10:15:00 | 781.05 | 787.60 | 781.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 781.05 | 787.60 | 781.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 781.05 | 787.60 | 781.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 779.50 | 785.98 | 781.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:15:00 | 782.85 | 785.98 | 781.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 789.80 | 787.15 | 782.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 795.65 | 800.30 | 800.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 795.65 | 800.30 | 800.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 791.00 | 798.44 | 799.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 10:15:00 | 796.95 | 794.35 | 796.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 10:15:00 | 796.95 | 794.35 | 796.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 796.95 | 794.35 | 796.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 796.95 | 794.35 | 796.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 783.90 | 792.26 | 795.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 780.05 | 792.26 | 795.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:45:00 | 781.25 | 782.77 | 788.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 801.95 | 790.06 | 788.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 801.95 | 790.06 | 788.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 804.50 | 792.95 | 790.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 799.40 | 808.92 | 803.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 799.40 | 808.92 | 803.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 799.40 | 808.92 | 803.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 799.40 | 808.92 | 803.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 807.00 | 808.53 | 803.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:15:00 | 807.55 | 808.53 | 803.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:00:00 | 807.15 | 806.51 | 804.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:45:00 | 812.95 | 807.99 | 806.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 848.50 | 852.96 | 853.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 848.50 | 852.96 | 853.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 838.25 | 847.64 | 850.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 822.35 | 818.05 | 827.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 822.35 | 818.05 | 827.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 810.00 | 802.82 | 809.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 815.95 | 802.82 | 809.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 818.15 | 805.88 | 810.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 824.65 | 805.88 | 810.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 816.65 | 808.04 | 810.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 820.50 | 808.04 | 810.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 820.15 | 812.40 | 812.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 14:15:00 | 823.60 | 816.01 | 814.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 10:15:00 | 815.35 | 817.22 | 815.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 10:15:00 | 815.35 | 817.22 | 815.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 815.35 | 817.22 | 815.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:30:00 | 819.00 | 818.18 | 816.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 811.05 | 817.37 | 816.08 | SL hit (close<static) qty=1.00 sl=814.05 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 811.70 | 815.22 | 815.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 802.00 | 811.10 | 813.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 810.00 | 808.59 | 811.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 810.00 | 808.59 | 811.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 810.00 | 808.59 | 811.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 806.90 | 808.59 | 811.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 808.45 | 807.96 | 810.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 808.45 | 807.96 | 810.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 811.95 | 808.59 | 810.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 811.95 | 808.59 | 810.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 813.00 | 809.47 | 810.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 820.30 | 809.47 | 810.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 821.00 | 811.78 | 811.41 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 810.40 | 813.96 | 814.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 807.10 | 812.23 | 813.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 11:15:00 | 810.60 | 810.20 | 811.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 11:15:00 | 810.60 | 810.20 | 811.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 810.60 | 810.20 | 811.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 810.60 | 810.20 | 811.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 804.80 | 809.12 | 811.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 803.05 | 809.12 | 811.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 802.50 | 808.71 | 810.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:45:00 | 803.00 | 805.62 | 808.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 803.90 | 805.85 | 808.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 807.90 | 806.26 | 808.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 807.50 | 806.26 | 808.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 806.00 | 806.21 | 807.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 813.30 | 808.74 | 808.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 813.30 | 808.74 | 808.55 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 803.00 | 807.70 | 808.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 799.45 | 806.05 | 807.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 809.75 | 805.97 | 807.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 809.75 | 805.97 | 807.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 809.75 | 805.97 | 807.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:00:00 | 802.60 | 805.29 | 806.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 14:15:00 | 800.50 | 800.30 | 802.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 15:15:00 | 803.95 | 801.25 | 802.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 827.00 | 806.83 | 804.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 827.00 | 806.83 | 804.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 829.10 | 811.28 | 806.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 863.00 | 863.57 | 849.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:15:00 | 856.95 | 863.57 | 849.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 863.00 | 867.20 | 859.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 875.55 | 867.20 | 859.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 858.20 | 865.38 | 863.92 | SL hit (close<static) qty=1.00 sl=858.40 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 852.25 | 862.75 | 862.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 841.65 | 856.82 | 860.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 796.85 | 794.70 | 807.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 797.10 | 794.70 | 807.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 814.60 | 797.38 | 799.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 813.90 | 797.38 | 799.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 810.15 | 799.93 | 800.41 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 814.55 | 802.86 | 801.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 815.95 | 807.50 | 804.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 810.00 | 810.89 | 807.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:00:00 | 810.00 | 810.89 | 807.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 810.05 | 810.90 | 807.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 811.10 | 810.90 | 807.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 811.95 | 811.48 | 808.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 818.30 | 811.48 | 808.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 801.10 | 809.00 | 808.29 | SL hit (close<static) qty=1.00 sl=804.05 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 798.40 | 806.88 | 807.39 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 807.15 | 804.97 | 804.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 823.50 | 808.67 | 806.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 815.00 | 815.03 | 810.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 815.00 | 815.03 | 810.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 816.90 | 818.96 | 815.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 816.90 | 818.96 | 815.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 824.70 | 828.86 | 824.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 824.70 | 828.86 | 824.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 823.20 | 827.73 | 824.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 823.20 | 827.73 | 824.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 827.05 | 827.59 | 824.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:45:00 | 829.10 | 827.88 | 825.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:00:00 | 830.80 | 828.47 | 825.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 834.75 | 839.20 | 839.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 834.75 | 839.20 | 839.58 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 849.90 | 840.98 | 840.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 852.95 | 846.06 | 843.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 838.05 | 846.94 | 845.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 14:15:00 | 838.05 | 846.94 | 845.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 838.05 | 846.94 | 845.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 838.05 | 846.94 | 845.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 840.00 | 845.55 | 845.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 847.15 | 845.55 | 845.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 15:15:00 | 873.55 | 878.92 | 878.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 15:15:00 | 873.55 | 878.92 | 878.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 869.80 | 874.70 | 876.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 871.25 | 860.42 | 863.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 871.25 | 860.42 | 863.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 871.25 | 860.42 | 863.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 871.25 | 860.42 | 863.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 870.65 | 862.47 | 864.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 870.65 | 862.47 | 864.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 870.05 | 866.38 | 865.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 873.00 | 868.42 | 867.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 883.45 | 888.79 | 882.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 883.45 | 888.79 | 882.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 883.45 | 888.79 | 882.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 882.30 | 888.79 | 882.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 878.75 | 886.08 | 882.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 878.75 | 886.08 | 882.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 876.00 | 884.07 | 881.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 882.00 | 884.07 | 881.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 882.55 | 883.76 | 881.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 14:45:00 | 885.95 | 885.07 | 882.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-23 09:15:00 | 974.55 | 950.15 | 932.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 921.05 | 934.17 | 935.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 914.45 | 926.18 | 930.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 870.40 | 863.74 | 876.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 15:00:00 | 870.40 | 863.74 | 876.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 872.00 | 865.40 | 876.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 880.65 | 865.40 | 876.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 885.65 | 869.45 | 877.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 890.50 | 869.45 | 877.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 873.60 | 870.28 | 877.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 868.95 | 869.54 | 876.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 872.75 | 870.42 | 874.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 884.35 | 874.96 | 874.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 884.35 | 874.96 | 874.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 886.65 | 877.30 | 875.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 873.00 | 876.44 | 875.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 12:15:00 | 873.00 | 876.44 | 875.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 873.00 | 876.44 | 875.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 873.00 | 876.44 | 875.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 874.40 | 876.03 | 875.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 14:30:00 | 877.50 | 877.27 | 875.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 15:00:00 | 882.20 | 877.27 | 875.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 868.15 | 877.12 | 875.97 | SL hit (close<static) qty=1.00 sl=872.05 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 868.95 | 874.05 | 874.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 866.15 | 871.82 | 873.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 863.50 | 862.67 | 867.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:45:00 | 862.65 | 862.67 | 867.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 863.65 | 857.96 | 862.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 863.65 | 857.96 | 862.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 870.15 | 860.40 | 863.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 870.15 | 860.40 | 863.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 870.60 | 862.44 | 864.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 869.90 | 862.44 | 864.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 892.85 | 869.90 | 867.37 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 868.00 | 875.64 | 876.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 866.00 | 873.71 | 875.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 864.35 | 860.39 | 864.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 12:15:00 | 864.35 | 860.39 | 864.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 864.35 | 860.39 | 864.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 865.15 | 860.39 | 864.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 864.40 | 861.19 | 864.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 864.40 | 861.19 | 864.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 860.55 | 861.06 | 864.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:45:00 | 864.00 | 861.06 | 864.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 865.00 | 862.00 | 863.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 863.40 | 862.00 | 863.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 870.40 | 863.68 | 864.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 870.40 | 863.68 | 864.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 865.80 | 864.10 | 864.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:15:00 | 873.80 | 864.10 | 864.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 877.70 | 866.82 | 865.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 882.85 | 871.83 | 868.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 10:15:00 | 879.60 | 883.72 | 878.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 879.60 | 883.72 | 878.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 879.60 | 883.72 | 878.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 879.60 | 883.72 | 878.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 885.90 | 884.16 | 879.50 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 862.60 | 876.42 | 877.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 858.25 | 872.79 | 875.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 878.80 | 870.38 | 873.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 878.80 | 870.38 | 873.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 878.80 | 870.38 | 873.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 878.80 | 870.38 | 873.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 870.10 | 870.33 | 873.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 867.00 | 870.33 | 873.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 14:15:00 | 876.55 | 864.59 | 864.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 14:15:00 | 876.55 | 864.59 | 864.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 879.60 | 868.55 | 866.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 862.55 | 872.01 | 869.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 862.55 | 872.01 | 869.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 862.55 | 872.01 | 869.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 862.55 | 872.01 | 869.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 859.75 | 869.56 | 868.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 859.50 | 869.56 | 868.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 861.55 | 867.96 | 868.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 853.65 | 862.78 | 865.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 832.05 | 828.44 | 838.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 832.05 | 828.44 | 838.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 832.05 | 828.44 | 838.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 810.60 | 830.65 | 834.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:45:00 | 822.25 | 821.37 | 825.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 822.05 | 820.45 | 820.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 822.05 | 820.45 | 820.39 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 819.60 | 820.25 | 820.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 813.95 | 818.80 | 819.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 823.90 | 818.73 | 819.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 823.90 | 818.73 | 819.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 823.90 | 818.73 | 819.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 823.90 | 818.73 | 819.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 824.05 | 819.79 | 819.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 824.00 | 819.79 | 819.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 831.65 | 822.17 | 820.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 849.20 | 830.25 | 825.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 863.40 | 863.40 | 853.82 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 12:30:00 | 868.20 | 865.13 | 857.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 14:45:00 | 868.00 | 865.79 | 858.78 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 15:15:00 | 868.00 | 865.05 | 861.78 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 868.00 | 865.64 | 862.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 918.15 | 865.64 | 862.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 911.61 | 874.42 | 866.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 911.40 | 874.42 | 866.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 911.40 | 874.42 | 866.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 888.30 | 895.48 | 882.54 | SL hit (close<ema200) qty=0.50 sl=895.48 alert=retest1 |

### Cycle 44 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 880.00 | 890.98 | 892.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 876.45 | 884.47 | 888.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 877.40 | 875.55 | 880.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 877.40 | 875.55 | 880.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 875.75 | 868.11 | 870.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 875.75 | 868.11 | 870.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 877.50 | 869.98 | 871.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:15:00 | 876.25 | 869.98 | 871.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 880.00 | 873.69 | 873.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 882.20 | 876.76 | 874.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 875.60 | 876.53 | 874.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 11:15:00 | 875.60 | 876.53 | 874.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 875.60 | 876.53 | 874.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 875.60 | 876.53 | 874.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 877.90 | 876.80 | 875.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:00:00 | 880.20 | 877.04 | 875.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:30:00 | 880.70 | 877.64 | 876.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:30:00 | 881.60 | 878.85 | 876.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 882.75 | 889.27 | 889.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 882.75 | 889.27 | 889.33 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 890.35 | 889.48 | 889.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 892.95 | 890.27 | 889.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 12:15:00 | 890.65 | 896.64 | 894.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 12:15:00 | 890.65 | 896.64 | 894.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 890.65 | 896.64 | 894.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 890.65 | 896.64 | 894.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 892.50 | 895.82 | 893.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 892.50 | 895.82 | 893.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 894.20 | 895.49 | 893.94 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 877.95 | 891.93 | 892.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 861.45 | 880.16 | 885.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 864.05 | 863.94 | 872.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 864.05 | 863.94 | 872.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 864.05 | 863.94 | 872.50 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 887.15 | 876.16 | 875.31 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 14:15:00 | 863.00 | 874.00 | 874.96 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 887.00 | 877.18 | 876.13 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 862.70 | 878.41 | 879.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 858.95 | 874.52 | 877.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 848.15 | 842.32 | 852.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 848.15 | 842.32 | 852.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 848.15 | 842.32 | 852.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 851.50 | 842.32 | 852.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 839.85 | 841.36 | 846.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 833.85 | 840.09 | 845.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 834.90 | 838.75 | 844.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 838.65 | 827.28 | 825.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 838.65 | 827.28 | 825.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 847.85 | 834.91 | 829.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 875.00 | 876.54 | 861.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:15:00 | 867.85 | 876.54 | 861.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 861.55 | 873.54 | 861.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 859.35 | 873.54 | 861.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 875.00 | 873.83 | 862.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 875.00 | 873.83 | 862.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 851.00 | 869.45 | 862.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:45:00 | 853.95 | 869.45 | 862.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 861.00 | 867.76 | 862.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 865.20 | 867.25 | 862.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 863.20 | 865.30 | 862.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 865.15 | 865.44 | 862.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 846.80 | 861.71 | 861.03 | SL hit (close<static) qty=1.00 sl=850.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 849.25 | 859.22 | 859.96 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 876.00 | 861.36 | 860.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 926.00 | 874.29 | 866.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 13:15:00 | 921.20 | 923.35 | 906.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 13:45:00 | 920.00 | 923.35 | 906.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 911.50 | 921.82 | 916.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 911.50 | 921.82 | 916.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 911.25 | 919.70 | 915.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 907.55 | 919.70 | 915.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 916.25 | 916.83 | 915.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 912.00 | 916.83 | 915.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 919.70 | 917.40 | 915.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 929.90 | 917.92 | 916.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 924.90 | 920.56 | 918.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 12:15:00 | 913.95 | 921.03 | 920.00 | SL hit (close<static) qty=1.00 sl=915.10 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 909.80 | 918.79 | 919.07 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 930.85 | 919.85 | 919.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 937.05 | 923.29 | 920.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 936.30 | 954.22 | 945.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 936.30 | 954.22 | 945.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 936.30 | 954.22 | 945.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 938.75 | 954.22 | 945.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 939.85 | 951.35 | 945.01 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 926.00 | 940.26 | 941.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 915.75 | 935.36 | 938.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 934.85 | 929.77 | 933.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 934.85 | 929.77 | 933.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 934.85 | 929.77 | 933.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 940.70 | 929.77 | 933.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 929.85 | 929.79 | 933.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:30:00 | 934.45 | 929.79 | 933.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 924.95 | 927.25 | 930.96 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 942.00 | 932.50 | 932.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 946.90 | 935.38 | 933.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 960.00 | 964.57 | 954.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 960.00 | 964.57 | 954.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 957.20 | 963.10 | 954.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 956.00 | 963.10 | 954.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 948.20 | 960.12 | 953.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:45:00 | 947.85 | 960.12 | 953.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 942.85 | 956.67 | 952.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 942.85 | 956.67 | 952.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 951.75 | 952.29 | 951.53 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 949.10 | 950.97 | 951.02 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 956.10 | 951.78 | 951.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 962.15 | 954.08 | 952.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 972.60 | 975.62 | 970.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 972.60 | 975.62 | 970.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 972.60 | 975.62 | 970.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 971.75 | 975.62 | 970.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 964.95 | 974.79 | 971.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 964.95 | 974.79 | 971.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 962.70 | 972.37 | 970.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 965.75 | 972.37 | 970.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 964.65 | 971.42 | 970.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 964.65 | 971.42 | 970.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 10:15:00 | 954.95 | 968.13 | 969.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 944.60 | 963.42 | 967.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 866.00 | 859.50 | 875.71 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:00:00 | 852.15 | 858.02 | 872.22 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 12:00:00 | 851.25 | 856.66 | 870.32 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 09:15:00 | 818.20 | 858.14 | 866.71 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 850.50 | 839.83 | 849.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 850.50 | 839.83 | 849.14 | SL hit (close>ema400) qty=1.00 sl=849.14 alert=retest1 |

### Cycle 63 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 879.50 | 858.50 | 855.88 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 838.60 | 857.87 | 859.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 829.10 | 845.80 | 851.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 832.10 | 829.83 | 838.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:45:00 | 835.40 | 829.83 | 838.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 832.20 | 826.81 | 833.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 832.90 | 826.81 | 833.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 844.35 | 830.66 | 834.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 844.35 | 830.66 | 834.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 848.25 | 834.18 | 835.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 848.25 | 834.18 | 835.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 850.50 | 837.44 | 836.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 852.40 | 842.16 | 839.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 855.20 | 870.39 | 859.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 855.20 | 870.39 | 859.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 855.20 | 870.39 | 859.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 854.50 | 870.39 | 859.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 859.50 | 868.22 | 859.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 857.10 | 868.22 | 859.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 860.55 | 866.68 | 859.96 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 850.00 | 857.78 | 857.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 845.75 | 855.37 | 856.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 826.95 | 824.52 | 833.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 826.95 | 824.52 | 833.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 841.00 | 828.98 | 833.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 841.55 | 828.98 | 833.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 850.70 | 833.33 | 835.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 850.70 | 833.33 | 835.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 846.00 | 838.54 | 837.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 867.45 | 844.32 | 840.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 852.00 | 859.68 | 852.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 852.00 | 859.68 | 852.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 852.00 | 859.68 | 852.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 852.00 | 859.68 | 852.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 846.40 | 857.03 | 851.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 846.40 | 857.03 | 851.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 843.00 | 854.22 | 850.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 843.00 | 854.22 | 850.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 845.45 | 851.85 | 850.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 845.45 | 851.85 | 850.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 833.05 | 848.09 | 848.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 830.30 | 844.53 | 847.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 833.45 | 820.30 | 829.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 833.45 | 820.30 | 829.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 833.45 | 820.30 | 829.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 825.60 | 824.31 | 830.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 850.60 | 829.56 | 831.98 | SL hit (close>static) qty=1.00 sl=845.75 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 869.80 | 840.25 | 836.58 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 831.05 | 835.59 | 835.71 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 895.00 | 843.32 | 836.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 907.00 | 856.05 | 843.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 889.60 | 890.85 | 870.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 889.60 | 890.85 | 870.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 917.65 | 928.49 | 918.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:00:00 | 917.65 | 928.49 | 918.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 916.00 | 925.99 | 917.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 916.00 | 925.99 | 917.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 917.00 | 924.19 | 917.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 935.55 | 924.19 | 917.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 11:15:00 | 1029.11 | 1012.45 | 998.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 997.00 | 1028.90 | 1032.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 983.05 | 1014.70 | 1024.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 995.40 | 994.16 | 1006.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 995.40 | 994.16 | 1006.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1057.00 | 1006.14 | 1008.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:45:00 | 1059.10 | 1006.14 | 1008.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 1045.40 | 1013.99 | 1011.57 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 992.75 | 1016.47 | 1019.03 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1032.10 | 1019.42 | 1017.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 1051.30 | 1026.42 | 1021.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 1096.60 | 1099.96 | 1085.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:00:00 | 1096.60 | 1099.96 | 1085.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-23 13:15:00 | 720.10 | 2025-05-27 12:15:00 | 736.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-05-27 09:15:00 | 721.00 | 2025-05-27 12:15:00 | 736.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-06-26 12:15:00 | 782.85 | 2025-07-04 10:15:00 | 795.65 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-06-26 12:45:00 | 789.80 | 2025-07-04 10:15:00 | 795.65 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-07-07 12:15:00 | 780.05 | 2025-07-09 12:15:00 | 801.95 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-08 10:45:00 | 781.25 | 2025-07-09 12:15:00 | 801.95 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-07-11 12:15:00 | 807.55 | 2025-07-23 11:15:00 | 848.50 | STOP_HIT | 1.00 | 5.07% |
| BUY | retest2 | 2025-07-14 10:00:00 | 807.15 | 2025-07-23 11:15:00 | 848.50 | STOP_HIT | 1.00 | 5.12% |
| BUY | retest2 | 2025-07-14 14:45:00 | 812.95 | 2025-07-23 11:15:00 | 848.50 | STOP_HIT | 1.00 | 4.37% |
| BUY | retest2 | 2025-07-31 12:30:00 | 819.00 | 2025-07-31 14:15:00 | 811.05 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-08-08 13:15:00 | 803.05 | 2025-08-12 11:15:00 | 813.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-08 14:15:00 | 802.50 | 2025-08-12 11:15:00 | 813.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-11 09:45:00 | 803.00 | 2025-08-12 11:15:00 | 813.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-11 11:30:00 | 803.90 | 2025-08-12 11:15:00 | 813.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-08-13 11:00:00 | 802.60 | 2025-08-18 09:15:00 | 827.00 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-08-14 14:15:00 | 800.50 | 2025-08-18 09:15:00 | 827.00 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-08-14 15:15:00 | 803.95 | 2025-08-18 09:15:00 | 827.00 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-08-22 09:15:00 | 875.55 | 2025-08-25 11:15:00 | 858.20 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-09-05 09:15:00 | 818.30 | 2025-09-05 11:15:00 | 801.10 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-09-15 13:45:00 | 829.10 | 2025-09-19 14:15:00 | 834.75 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2025-09-15 15:00:00 | 830.80 | 2025-09-19 14:15:00 | 834.75 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-09-24 09:15:00 | 847.15 | 2025-10-06 15:15:00 | 873.55 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest2 | 2025-10-15 14:45:00 | 885.95 | 2025-10-23 09:15:00 | 974.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-31 11:45:00 | 868.95 | 2025-11-04 10:15:00 | 884.35 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-11-03 09:15:00 | 872.75 | 2025-11-04 10:15:00 | 884.35 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-11-04 14:30:00 | 877.50 | 2025-11-06 09:15:00 | 868.15 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-11-04 15:00:00 | 882.20 | 2025-11-06 09:15:00 | 868.15 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-11-25 09:15:00 | 867.00 | 2025-11-27 14:15:00 | 876.55 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-05 10:15:00 | 810.60 | 2025-12-10 10:15:00 | 822.05 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-08 11:45:00 | 822.25 | 2025-12-10 10:15:00 | 822.05 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest1 | 2025-12-16 12:30:00 | 868.20 | 2025-12-18 09:15:00 | 911.61 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-16 14:45:00 | 868.00 | 2025-12-18 09:15:00 | 911.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-17 15:15:00 | 868.00 | 2025-12-18 09:15:00 | 911.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-16 12:30:00 | 868.20 | 2025-12-18 15:15:00 | 888.30 | STOP_HIT | 0.50 | 2.32% |
| BUY | retest1 | 2025-12-16 14:45:00 | 868.00 | 2025-12-18 15:15:00 | 888.30 | STOP_HIT | 0.50 | 2.34% |
| BUY | retest1 | 2025-12-17 15:15:00 | 868.00 | 2025-12-18 15:15:00 | 888.30 | STOP_HIT | 0.50 | 2.34% |
| BUY | retest2 | 2025-12-18 09:15:00 | 918.15 | 2025-12-24 14:15:00 | 880.00 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2026-01-02 11:00:00 | 880.20 | 2026-01-07 10:15:00 | 882.75 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2026-01-02 11:30:00 | 880.70 | 2026-01-07 10:15:00 | 882.75 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2026-01-02 12:30:00 | 881.60 | 2026-01-07 10:15:00 | 882.75 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2026-01-23 10:45:00 | 833.85 | 2026-01-29 09:15:00 | 838.65 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-23 11:30:00 | 834.90 | 2026-01-29 09:15:00 | 838.65 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-02-01 15:00:00 | 865.20 | 2026-02-02 10:15:00 | 846.80 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-02-02 09:15:00 | 863.20 | 2026-02-02 10:15:00 | 846.80 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-02 09:45:00 | 865.15 | 2026-02-02 10:15:00 | 846.80 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-02-02 10:30:00 | 863.25 | 2026-02-02 11:15:00 | 849.25 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-02-09 09:15:00 | 929.90 | 2026-02-10 12:15:00 | 913.95 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-10 09:15:00 | 924.90 | 2026-02-10 12:15:00 | 913.95 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest1 | 2026-03-06 11:00:00 | 852.15 | 2026-03-10 09:15:00 | 850.50 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest1 | 2026-03-06 12:00:00 | 851.25 | 2026-03-10 09:15:00 | 850.50 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest1 | 2026-03-09 09:15:00 | 818.20 | 2026-03-10 09:15:00 | 850.50 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2026-04-01 11:30:00 | 825.60 | 2026-04-01 12:15:00 | 850.60 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2026-04-15 09:15:00 | 935.55 | 2026-04-21 11:15:00 | 1029.11 | TARGET_HIT | 1.00 | 10.00% |
