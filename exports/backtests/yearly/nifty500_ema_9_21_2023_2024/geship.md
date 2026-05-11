# Great Eastern Shipping Co. Ltd. (GESHIP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1589.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 156 |
| ALERT2 | 152 |
| ALERT2_SKIP | 81 |
| ALERT3 | 379 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 183 |
| PARTIAL | 10 |
| TARGET_HIT | 12 |
| STOP_HIT | 176 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 197 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 49 / 148
- **Target hits / Stop hits / Partials:** 12 / 175 / 10
- **Avg / median % per leg:** -0.26% / -1.22%
- **Sum % (uncompounded):** -51.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 98 | 26 | 26.5% | 12 | 85 | 1 | 0.48% | 47.0% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.25% | 7.5% |
| BUY @ 3rd Alert (retest2) | 92 | 23 | 25.0% | 12 | 80 | 0 | 0.43% | 39.5% |
| SELL (all) | 99 | 23 | 23.2% | 0 | 90 | 9 | -1.00% | -98.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.59% | -2.6% |
| SELL @ 3rd Alert (retest2) | 98 | 23 | 23.5% | 0 | 89 | 9 | -0.98% | -96.3% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 6 | 1 | 0.70% | 4.9% |
| retest2 (combined) | 190 | 46 | 24.2% | 12 | 169 | 9 | -0.30% | -56.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 10:15:00 | 689.55 | 702.98 | 704.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 14:15:00 | 684.50 | 695.23 | 699.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 12:15:00 | 678.55 | 670.50 | 676.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 12:15:00 | 678.55 | 670.50 | 676.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 678.55 | 670.50 | 676.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:00:00 | 678.55 | 670.50 | 676.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 671.95 | 670.79 | 675.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 12:30:00 | 670.20 | 672.60 | 675.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 09:15:00 | 663.50 | 672.03 | 674.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-25 10:15:00 | 681.75 | 673.22 | 674.29 | SL hit (close>static) qty=1.00 sl=679.80 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 12:15:00 | 678.00 | 675.32 | 675.13 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 09:15:00 | 667.40 | 674.32 | 674.84 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 13:15:00 | 678.85 | 674.50 | 674.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 14:15:00 | 684.50 | 676.50 | 675.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 15:15:00 | 683.00 | 686.27 | 682.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 15:15:00 | 683.00 | 686.27 | 682.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 15:15:00 | 683.00 | 686.27 | 682.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:15:00 | 676.05 | 686.27 | 682.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 681.40 | 685.30 | 682.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 13:15:00 | 689.65 | 684.31 | 682.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 15:15:00 | 686.95 | 685.45 | 683.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 11:30:00 | 686.10 | 686.07 | 684.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 15:15:00 | 690.50 | 683.89 | 683.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 690.50 | 685.21 | 684.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-05 10:15:00 | 677.70 | 683.45 | 684.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 10:15:00 | 677.70 | 683.45 | 684.09 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 13:15:00 | 717.80 | 689.90 | 686.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-09 09:15:00 | 724.85 | 715.30 | 710.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-12 11:15:00 | 718.25 | 721.25 | 717.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 11:15:00 | 718.25 | 721.25 | 717.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 718.25 | 721.25 | 717.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:45:00 | 718.70 | 721.25 | 717.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 732.25 | 723.45 | 718.80 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 12:15:00 | 718.10 | 720.48 | 720.59 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 15:15:00 | 721.05 | 720.67 | 720.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 09:15:00 | 759.40 | 728.42 | 724.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 14:15:00 | 755.95 | 759.09 | 749.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 15:00:00 | 755.95 | 759.09 | 749.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 743.15 | 755.57 | 749.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 10:00:00 | 743.15 | 755.57 | 749.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 743.00 | 753.05 | 748.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 11:00:00 | 743.00 | 753.05 | 748.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 15:15:00 | 750.95 | 749.02 | 748.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:15:00 | 745.40 | 749.02 | 748.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 742.30 | 747.67 | 747.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:30:00 | 743.80 | 747.67 | 747.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 728.95 | 743.93 | 745.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 14:15:00 | 724.25 | 733.15 | 739.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 736.95 | 732.77 | 738.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 736.95 | 732.77 | 738.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 736.95 | 732.77 | 738.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:45:00 | 738.10 | 732.77 | 738.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 733.60 | 732.93 | 737.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 14:30:00 | 729.00 | 733.54 | 736.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 09:15:00 | 692.55 | 718.71 | 726.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-26 09:15:00 | 723.20 | 709.86 | 716.47 | SL hit (close>ema200) qty=0.50 sl=709.86 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 734.00 | 720.27 | 719.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 09:15:00 | 747.30 | 725.68 | 721.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 09:15:00 | 744.65 | 746.45 | 740.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 15:15:00 | 738.35 | 743.85 | 741.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 15:15:00 | 738.35 | 743.85 | 741.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 09:15:00 | 735.05 | 743.85 | 741.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 731.00 | 741.28 | 740.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 09:30:00 | 730.70 | 741.28 | 740.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 10:15:00 | 731.00 | 739.22 | 739.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 09:15:00 | 720.50 | 731.71 | 735.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 732.85 | 728.46 | 731.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 732.85 | 728.46 | 731.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 732.85 | 728.46 | 731.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:45:00 | 732.00 | 728.46 | 731.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 731.95 | 729.16 | 731.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:30:00 | 734.60 | 729.16 | 731.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 11:15:00 | 732.00 | 729.73 | 731.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 12:45:00 | 727.25 | 729.57 | 731.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 11:15:00 | 734.90 | 730.01 | 730.58 | SL hit (close>static) qty=1.00 sl=732.15 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 12:15:00 | 737.40 | 731.49 | 731.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 14:15:00 | 740.80 | 734.16 | 732.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 733.45 | 735.19 | 733.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 733.45 | 735.19 | 733.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 733.45 | 735.19 | 733.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:15:00 | 732.50 | 735.19 | 733.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 734.80 | 735.11 | 733.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 11:30:00 | 736.60 | 735.51 | 733.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 10:45:00 | 735.40 | 739.32 | 736.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-14 09:15:00 | 810.26 | 771.94 | 765.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 12:15:00 | 777.90 | 786.12 | 786.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 11:15:00 | 772.25 | 780.63 | 783.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 11:15:00 | 774.50 | 774.47 | 778.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-21 12:15:00 | 771.40 | 774.47 | 778.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 782.95 | 772.67 | 775.61 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 786.85 | 778.04 | 777.19 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 13:15:00 | 774.50 | 777.52 | 777.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 14:15:00 | 767.55 | 775.53 | 776.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 776.55 | 775.17 | 776.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 776.55 | 775.17 | 776.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 776.55 | 775.17 | 776.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:15:00 | 773.00 | 776.12 | 776.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-31 11:15:00 | 778.50 | 769.03 | 769.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 778.50 | 769.03 | 769.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 13:15:00 | 785.00 | 773.79 | 771.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 13:15:00 | 789.20 | 791.18 | 786.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 13:45:00 | 791.90 | 791.18 | 786.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 790.85 | 791.11 | 786.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 797.80 | 790.63 | 786.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 12:15:00 | 783.40 | 796.97 | 798.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 12:15:00 | 783.40 | 796.97 | 798.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 13:15:00 | 774.00 | 792.37 | 796.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 15:15:00 | 758.00 | 757.68 | 765.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-10 09:15:00 | 754.80 | 757.68 | 765.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 758.30 | 757.81 | 764.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 10:30:00 | 752.60 | 757.04 | 763.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 10:15:00 | 766.55 | 760.93 | 762.47 | SL hit (close>static) qty=1.00 sl=766.05 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 786.45 | 756.96 | 753.52 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 14:15:00 | 767.05 | 772.13 | 772.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 15:15:00 | 765.80 | 770.87 | 771.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 09:15:00 | 777.55 | 772.20 | 772.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 777.55 | 772.20 | 772.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 777.55 | 772.20 | 772.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:00:00 | 777.55 | 772.20 | 772.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 10:15:00 | 774.00 | 772.56 | 772.43 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 12:15:00 | 771.70 | 772.30 | 772.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 13:15:00 | 770.15 | 771.87 | 772.13 | Break + close below crossover candle low |

### Cycle 22 — BUY (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 14:15:00 | 778.70 | 773.24 | 772.73 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 767.50 | 773.73 | 774.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 759.35 | 769.74 | 772.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 15:15:00 | 765.25 | 764.17 | 767.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 09:15:00 | 765.60 | 764.17 | 767.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 765.30 | 764.39 | 767.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 14:00:00 | 759.95 | 763.79 | 766.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 11:30:00 | 756.00 | 761.16 | 764.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 10:15:00 | 759.00 | 757.54 | 760.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 11:15:00 | 758.50 | 758.23 | 760.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 11:15:00 | 758.45 | 758.28 | 760.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 12:15:00 | 758.95 | 758.28 | 760.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 761.00 | 758.82 | 760.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 12:30:00 | 760.75 | 758.82 | 760.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 757.10 | 758.48 | 760.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-31 15:15:00 | 766.00 | 760.93 | 760.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 15:15:00 | 766.00 | 760.93 | 760.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 09:15:00 | 780.15 | 764.78 | 762.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 14:15:00 | 769.55 | 769.99 | 766.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 14:15:00 | 769.55 | 769.99 | 766.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 769.55 | 769.99 | 766.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:45:00 | 767.60 | 769.99 | 766.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 771.95 | 770.38 | 766.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 10:00:00 | 773.70 | 771.05 | 767.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 13:45:00 | 772.65 | 772.07 | 769.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 10:00:00 | 772.85 | 771.14 | 769.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 11:45:00 | 772.40 | 771.44 | 769.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 770.40 | 771.23 | 769.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:30:00 | 769.00 | 771.23 | 769.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 767.80 | 770.54 | 769.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 14:00:00 | 767.80 | 770.54 | 769.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 14:15:00 | 769.00 | 770.23 | 769.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 14:30:00 | 767.15 | 770.23 | 769.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 15:15:00 | 768.80 | 769.95 | 769.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 09:15:00 | 770.15 | 769.95 | 769.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 10:30:00 | 771.60 | 769.98 | 769.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 11:00:00 | 770.90 | 769.98 | 769.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-07 11:15:00 | 849.92 | 799.97 | 784.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 809.80 | 817.37 | 817.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 800.10 | 813.91 | 815.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 813.15 | 811.22 | 813.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 10:15:00 | 813.15 | 811.22 | 813.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 813.15 | 811.22 | 813.94 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 818.70 | 815.27 | 815.14 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 09:15:00 | 812.75 | 814.99 | 815.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-14 10:15:00 | 808.80 | 813.75 | 814.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 14:15:00 | 814.75 | 812.08 | 813.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 14:15:00 | 814.75 | 812.08 | 813.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 814.75 | 812.08 | 813.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 15:00:00 | 814.75 | 812.08 | 813.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 810.05 | 811.67 | 812.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:15:00 | 818.85 | 811.67 | 812.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 817.30 | 812.80 | 813.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:30:00 | 819.05 | 812.80 | 813.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 10:15:00 | 819.70 | 814.18 | 813.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 838.45 | 820.62 | 817.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 14:15:00 | 830.50 | 830.99 | 825.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 15:00:00 | 830.50 | 830.99 | 825.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 825.00 | 829.79 | 825.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 826.35 | 829.79 | 825.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 825.60 | 828.95 | 825.42 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 12:15:00 | 811.80 | 823.20 | 823.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 804.30 | 817.04 | 820.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 802.10 | 801.57 | 809.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 10:00:00 | 802.10 | 801.57 | 809.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 806.00 | 802.45 | 809.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:45:00 | 810.00 | 802.45 | 809.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 11:15:00 | 811.00 | 804.16 | 809.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 12:00:00 | 811.00 | 804.16 | 809.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 807.20 | 804.77 | 809.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 12:30:00 | 811.45 | 804.77 | 809.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 810.35 | 806.09 | 808.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 14:30:00 | 809.00 | 806.09 | 808.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 811.80 | 807.23 | 809.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 816.45 | 807.23 | 809.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 819.30 | 809.65 | 810.10 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 10:15:00 | 819.20 | 811.56 | 810.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 821.00 | 814.64 | 812.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 12:15:00 | 819.70 | 820.12 | 817.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 13:15:00 | 817.50 | 820.12 | 817.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 13:15:00 | 833.80 | 822.86 | 818.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 13:30:00 | 821.40 | 822.86 | 818.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 842.80 | 850.33 | 845.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 842.80 | 850.33 | 845.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 841.20 | 848.50 | 844.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:30:00 | 841.25 | 848.50 | 844.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 844.60 | 847.72 | 844.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:15:00 | 843.00 | 847.72 | 844.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 860.10 | 850.20 | 846.33 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 838.70 | 849.63 | 850.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 14:15:00 | 827.15 | 838.79 | 843.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 837.40 | 836.60 | 841.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 837.40 | 836.60 | 841.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 837.40 | 836.60 | 841.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:45:00 | 837.30 | 836.60 | 841.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 839.05 | 837.09 | 841.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 10:45:00 | 841.45 | 837.09 | 841.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 840.15 | 837.77 | 840.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 13:00:00 | 840.15 | 837.77 | 840.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 837.00 | 837.62 | 840.55 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 10:15:00 | 850.80 | 842.53 | 841.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 09:15:00 | 857.15 | 848.56 | 845.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-10 12:15:00 | 851.00 | 851.18 | 847.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 12:15:00 | 851.00 | 851.18 | 847.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 851.00 | 851.18 | 847.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:30:00 | 850.40 | 851.18 | 847.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 14:15:00 | 851.20 | 851.27 | 848.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-10 14:45:00 | 846.90 | 851.27 | 848.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 850.90 | 851.19 | 848.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:15:00 | 853.60 | 851.19 | 848.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 862.05 | 853.36 | 849.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 10:15:00 | 864.50 | 853.36 | 849.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 865.80 | 856.03 | 852.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 14:15:00 | 864.50 | 865.56 | 862.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 09:15:00 | 867.00 | 873.21 | 873.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 09:15:00 | 867.00 | 873.21 | 873.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 858.65 | 870.30 | 872.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 13:15:00 | 870.95 | 868.74 | 870.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 13:15:00 | 870.95 | 868.74 | 870.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 13:15:00 | 870.95 | 868.74 | 870.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 14:00:00 | 870.95 | 868.74 | 870.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 869.05 | 868.80 | 870.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 869.05 | 868.80 | 870.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 868.95 | 866.94 | 869.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 10:15:00 | 873.30 | 866.94 | 869.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 876.05 | 868.77 | 870.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:00:00 | 876.05 | 868.77 | 870.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 878.00 | 870.61 | 870.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 12:00:00 | 878.00 | 870.61 | 870.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 12:15:00 | 872.80 | 871.05 | 870.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 14:15:00 | 880.00 | 873.98 | 872.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 09:15:00 | 872.35 | 874.62 | 873.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 872.35 | 874.62 | 873.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 872.35 | 874.62 | 873.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:00:00 | 872.35 | 874.62 | 873.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 867.50 | 873.20 | 872.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 11:00:00 | 867.50 | 873.20 | 872.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 863.65 | 871.29 | 871.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 13:15:00 | 858.50 | 867.16 | 869.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 823.80 | 806.80 | 817.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 823.80 | 806.80 | 817.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 823.80 | 806.80 | 817.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:30:00 | 825.20 | 806.80 | 817.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 824.30 | 810.30 | 818.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 824.45 | 810.30 | 818.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 12:15:00 | 821.70 | 814.26 | 818.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 09:15:00 | 797.80 | 819.75 | 820.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 09:15:00 | 757.91 | 759.77 | 768.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-10 09:15:00 | 748.10 | 741.91 | 752.65 | SL hit (close>ema200) qty=0.50 sl=741.91 alert=retest2 |

### Cycle 36 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 778.05 | 757.55 | 756.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 788.40 | 774.75 | 767.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 09:15:00 | 785.20 | 786.27 | 778.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 10:00:00 | 785.20 | 786.27 | 778.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 11:15:00 | 780.50 | 785.29 | 779.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 11:45:00 | 780.00 | 785.29 | 779.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 783.00 | 784.14 | 780.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:30:00 | 786.00 | 784.43 | 780.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 10:00:00 | 785.60 | 784.43 | 780.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 10:45:00 | 786.55 | 784.74 | 781.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 14:15:00 | 779.40 | 782.77 | 781.46 | SL hit (close<static) qty=1.00 sl=780.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 12:15:00 | 780.15 | 780.81 | 780.84 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 13:15:00 | 783.20 | 781.29 | 781.06 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 15:15:00 | 780.00 | 780.78 | 780.85 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 789.00 | 782.43 | 781.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 11:15:00 | 793.50 | 785.47 | 783.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 15:15:00 | 826.00 | 826.28 | 817.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 09:15:00 | 824.85 | 826.28 | 817.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 821.20 | 824.63 | 818.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:30:00 | 817.75 | 824.63 | 818.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 816.70 | 823.04 | 818.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 12:00:00 | 816.70 | 823.04 | 818.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 819.75 | 822.38 | 818.33 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 10:15:00 | 811.20 | 816.47 | 816.70 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 09:15:00 | 833.50 | 817.28 | 816.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 10:15:00 | 843.50 | 822.52 | 818.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 13:15:00 | 869.50 | 872.99 | 855.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 13:45:00 | 871.10 | 872.99 | 855.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 885.90 | 893.25 | 885.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:00:00 | 885.90 | 893.25 | 885.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 887.80 | 892.16 | 885.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:30:00 | 893.70 | 891.51 | 886.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 14:45:00 | 892.00 | 893.90 | 890.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 09:15:00 | 900.00 | 892.82 | 890.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 11:15:00 | 891.50 | 892.88 | 890.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 890.40 | 892.39 | 890.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-08 13:15:00 | 882.65 | 889.25 | 889.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 882.65 | 889.25 | 889.38 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 14:15:00 | 894.95 | 890.39 | 889.89 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 09:15:00 | 883.40 | 889.09 | 889.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 15:15:00 | 877.00 | 882.95 | 885.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 866.10 | 862.99 | 869.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 866.10 | 862.99 | 869.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 866.10 | 862.99 | 869.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 11:00:00 | 857.55 | 861.90 | 868.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 12:15:00 | 876.00 | 864.04 | 865.17 | SL hit (close>static) qty=1.00 sl=872.00 alert=retest2 |

### Cycle 46 — BUY (started 2023-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 13:15:00 | 891.00 | 869.43 | 867.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 14:15:00 | 896.85 | 874.92 | 870.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 973.45 | 977.64 | 954.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 973.45 | 977.64 | 954.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 973.45 | 977.64 | 954.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:30:00 | 970.35 | 977.64 | 954.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 982.90 | 975.79 | 958.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:30:00 | 994.35 | 982.12 | 966.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 11:15:00 | 987.10 | 995.04 | 995.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 11:15:00 | 987.10 | 995.04 | 995.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-27 13:15:00 | 982.70 | 991.12 | 993.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 10:15:00 | 984.00 | 976.63 | 982.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 10:15:00 | 984.00 | 976.63 | 982.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 984.00 | 976.63 | 982.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:30:00 | 981.90 | 976.63 | 982.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 991.95 | 979.69 | 983.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 12:00:00 | 991.95 | 979.69 | 983.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 978.20 | 980.10 | 982.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 09:45:00 | 986.45 | 980.10 | 982.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 982.10 | 980.25 | 981.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:00:00 | 982.10 | 980.25 | 981.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 983.00 | 980.80 | 981.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:45:00 | 982.40 | 980.80 | 981.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 975.20 | 979.68 | 981.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 15:15:00 | 970.00 | 979.68 | 981.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-02 09:45:00 | 965.75 | 975.37 | 978.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 12:15:00 | 986.95 | 976.87 | 978.63 | SL hit (close>static) qty=1.00 sl=983.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 09:15:00 | 988.85 | 980.74 | 980.10 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 14:15:00 | 974.80 | 984.94 | 985.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 13:15:00 | 971.00 | 978.53 | 982.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-08 15:15:00 | 971.00 | 970.44 | 974.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-09 09:15:00 | 978.50 | 970.44 | 974.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 974.50 | 971.25 | 974.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 09:30:00 | 972.25 | 971.25 | 974.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 969.40 | 970.88 | 974.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 14:30:00 | 966.05 | 969.22 | 972.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 11:15:00 | 974.05 | 969.42 | 968.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 11:15:00 | 974.05 | 969.42 | 968.92 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 10:15:00 | 965.00 | 968.30 | 968.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 14:15:00 | 960.30 | 965.48 | 967.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 12:15:00 | 960.85 | 960.22 | 963.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-15 13:00:00 | 960.85 | 960.22 | 963.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 971.95 | 960.79 | 962.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:30:00 | 973.45 | 960.79 | 962.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 971.85 | 963.00 | 963.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 11:00:00 | 971.85 | 963.00 | 963.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 11:15:00 | 975.00 | 965.40 | 964.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 14:15:00 | 975.60 | 968.92 | 966.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 969.85 | 970.16 | 967.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 969.85 | 970.16 | 967.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 969.85 | 970.16 | 967.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:45:00 | 968.30 | 970.16 | 967.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 979.80 | 972.09 | 968.57 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 14:15:00 | 953.15 | 964.97 | 966.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 944.45 | 958.79 | 963.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 14:15:00 | 959.15 | 958.03 | 960.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 14:15:00 | 959.15 | 958.03 | 960.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 14:15:00 | 959.15 | 958.03 | 960.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 15:00:00 | 959.15 | 958.03 | 960.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 954.85 | 957.39 | 960.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:15:00 | 967.65 | 957.39 | 960.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 968.00 | 959.51 | 961.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 10:45:00 | 957.80 | 958.58 | 960.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 14:15:00 | 967.65 | 962.17 | 961.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 967.65 | 962.17 | 961.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 09:15:00 | 978.00 | 965.75 | 963.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 972.00 | 972.45 | 968.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 15:00:00 | 972.00 | 972.45 | 968.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 968.50 | 971.66 | 968.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 977.50 | 971.66 | 968.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 981.55 | 973.64 | 969.39 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 959.50 | 967.41 | 967.86 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 972.00 | 967.79 | 967.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 1007.05 | 975.64 | 970.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 09:15:00 | 1010.55 | 1020.11 | 1007.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 1010.55 | 1020.11 | 1007.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 1010.55 | 1020.11 | 1007.19 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 990.00 | 1002.34 | 1002.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 11:15:00 | 981.80 | 995.56 | 999.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 09:15:00 | 995.85 | 992.24 | 995.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 995.85 | 992.24 | 995.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 995.85 | 992.24 | 995.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:30:00 | 995.85 | 992.24 | 995.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 991.35 | 992.06 | 995.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:45:00 | 994.45 | 992.06 | 995.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 981.40 | 987.59 | 991.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 10:30:00 | 975.00 | 985.50 | 990.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 12:30:00 | 972.05 | 980.89 | 987.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 14:15:00 | 976.95 | 980.44 | 983.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-06 11:15:00 | 1022.00 | 987.55 | 984.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 11:15:00 | 1022.00 | 987.55 | 984.89 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 960.20 | 990.09 | 994.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 936.70 | 961.67 | 975.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 940.65 | 938.68 | 954.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 12:15:00 | 952.85 | 941.73 | 952.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 952.85 | 941.73 | 952.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:00:00 | 952.85 | 941.73 | 952.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 944.00 | 942.18 | 951.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 14:15:00 | 943.00 | 942.18 | 951.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 11:00:00 | 941.55 | 941.01 | 947.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 10:15:00 | 965.00 | 946.83 | 946.97 | SL hit (close>static) qty=1.00 sl=954.10 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 963.05 | 950.07 | 948.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 981.15 | 959.35 | 953.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 956.00 | 960.17 | 955.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 11:15:00 | 956.00 | 960.17 | 955.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 956.00 | 960.17 | 955.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:45:00 | 955.85 | 960.17 | 955.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 954.15 | 958.97 | 955.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 12:45:00 | 954.05 | 958.97 | 955.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 949.15 | 957.00 | 954.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:45:00 | 951.15 | 957.00 | 954.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 946.50 | 954.90 | 953.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 15:00:00 | 946.50 | 954.90 | 953.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 09:15:00 | 930.50 | 948.98 | 951.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 13:15:00 | 925.50 | 938.23 | 944.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 12:15:00 | 932.00 | 931.19 | 937.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-20 12:30:00 | 933.35 | 931.19 | 937.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 928.75 | 930.63 | 936.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:45:00 | 940.00 | 930.63 | 936.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 936.95 | 932.58 | 936.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 10:15:00 | 932.25 | 932.58 | 936.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 14:00:00 | 934.00 | 928.65 | 930.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 09:15:00 | 948.45 | 930.66 | 930.80 | SL hit (close>static) qty=1.00 sl=943.10 alert=retest2 |

### Cycle 62 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 961.20 | 936.77 | 933.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 964.50 | 952.74 | 944.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 14:15:00 | 958.50 | 959.48 | 951.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 15:00:00 | 958.50 | 959.48 | 951.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 953.15 | 957.66 | 952.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 11:00:00 | 953.15 | 957.66 | 952.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 950.00 | 956.13 | 952.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:00:00 | 950.00 | 956.13 | 952.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 950.00 | 954.90 | 952.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:30:00 | 949.90 | 954.90 | 952.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 969.90 | 956.37 | 953.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 11:15:00 | 993.00 | 961.63 | 957.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 12:15:00 | 993.75 | 1002.53 | 1002.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 993.75 | 1002.53 | 1002.89 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 13:15:00 | 1011.00 | 1003.17 | 1002.52 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 973.55 | 998.49 | 1000.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 12:15:00 | 971.40 | 986.18 | 993.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 946.10 | 926.31 | 942.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 946.10 | 926.31 | 942.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 946.10 | 926.31 | 942.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:00:00 | 946.10 | 926.31 | 942.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 950.95 | 931.23 | 943.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:45:00 | 951.45 | 931.23 | 943.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 15:15:00 | 974.00 | 953.37 | 950.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 12:15:00 | 991.40 | 964.80 | 957.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 11:15:00 | 999.95 | 1003.45 | 990.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-19 11:30:00 | 998.70 | 1003.45 | 990.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 989.85 | 998.80 | 992.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 989.85 | 998.80 | 992.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 993.50 | 997.74 | 993.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 990.80 | 997.74 | 993.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 994.90 | 997.17 | 993.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:30:00 | 994.60 | 997.17 | 993.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 994.00 | 996.05 | 993.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 14:00:00 | 994.00 | 996.05 | 993.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 981.20 | 993.08 | 992.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 15:00:00 | 981.20 | 993.08 | 992.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 15:15:00 | 976.50 | 989.77 | 990.80 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 10:15:00 | 999.90 | 990.24 | 989.39 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 10:15:00 | 979.60 | 989.21 | 990.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 13:15:00 | 972.75 | 982.86 | 986.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 10:15:00 | 979.70 | 978.72 | 983.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 10:45:00 | 979.90 | 978.72 | 983.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 11:15:00 | 985.00 | 979.98 | 983.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:00:00 | 985.00 | 979.98 | 983.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 983.00 | 980.58 | 983.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:30:00 | 985.00 | 980.58 | 983.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 980.00 | 980.47 | 982.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:30:00 | 980.00 | 980.47 | 982.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 996.00 | 983.57 | 984.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 996.00 | 983.57 | 984.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 15:15:00 | 999.85 | 986.83 | 985.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 09:15:00 | 1018.00 | 1002.59 | 997.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 12:15:00 | 1004.50 | 1005.65 | 1000.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-02 13:00:00 | 1004.50 | 1005.65 | 1000.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 998.55 | 1004.05 | 1000.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 15:00:00 | 998.55 | 1004.05 | 1000.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 15:15:00 | 986.95 | 1000.63 | 999.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 09:45:00 | 1016.60 | 1005.51 | 1001.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 10:15:00 | 1000.05 | 1019.59 | 1020.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 10:15:00 | 1000.05 | 1019.59 | 1020.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 13:15:00 | 994.50 | 1008.60 | 1014.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 10:15:00 | 991.95 | 990.27 | 997.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-10 11:00:00 | 991.95 | 990.27 | 997.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 974.30 | 983.89 | 990.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 11:00:00 | 971.25 | 981.36 | 988.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 922.69 | 965.08 | 976.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 985.35 | 963.99 | 969.27 | SL hit (close>ema200) qty=0.50 sl=963.99 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 11:15:00 | 988.05 | 972.45 | 972.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 12:15:00 | 991.70 | 976.30 | 974.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 14:15:00 | 1007.70 | 1016.57 | 1006.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 14:15:00 | 1007.70 | 1016.57 | 1006.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 1007.70 | 1016.57 | 1006.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 15:00:00 | 1007.70 | 1016.57 | 1006.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 1005.50 | 1014.35 | 1006.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 1021.30 | 1014.35 | 1006.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-02 09:15:00 | 1123.43 | 1094.05 | 1090.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 12:15:00 | 1077.25 | 1088.55 | 1088.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 13:15:00 | 1069.75 | 1084.79 | 1086.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 995.00 | 991.64 | 1009.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 15:15:00 | 967.75 | 985.32 | 994.34 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 981.40 | 981.72 | 990.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:30:00 | 985.60 | 981.72 | 990.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 988.30 | 982.65 | 988.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:00:00 | 988.30 | 982.65 | 988.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 986.50 | 983.42 | 988.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:30:00 | 996.65 | 983.42 | 988.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 992.85 | 985.31 | 988.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-10 15:15:00 | 992.85 | 985.31 | 988.70 | SL hit (close>ema400) qty=1.00 sl=988.70 alert=retest1 |

### Cycle 74 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1002.00 | 992.66 | 991.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 11:15:00 | 1019.65 | 998.06 | 994.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 14:15:00 | 997.80 | 998.82 | 995.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 14:15:00 | 997.80 | 998.82 | 995.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 997.80 | 998.82 | 995.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:45:00 | 996.40 | 998.82 | 995.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1027.85 | 1054.53 | 1048.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:00:00 | 1027.85 | 1054.53 | 1048.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 1026.15 | 1048.85 | 1046.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:00:00 | 1026.15 | 1048.85 | 1046.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 11:15:00 | 1024.55 | 1043.99 | 1044.32 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 1067.40 | 1045.08 | 1043.17 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 09:15:00 | 1030.00 | 1042.66 | 1044.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 13:15:00 | 1025.00 | 1034.25 | 1039.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 1033.80 | 1031.50 | 1036.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 1033.80 | 1031.50 | 1036.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1033.80 | 1031.50 | 1036.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:45:00 | 1024.60 | 1030.06 | 1034.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 1059.00 | 1018.06 | 1016.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 1059.00 | 1018.06 | 1016.12 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 1026.00 | 1033.13 | 1033.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 11:15:00 | 1020.00 | 1030.50 | 1031.95 | Break + close below crossover candle low |

### Cycle 80 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 1050.50 | 1033.14 | 1032.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 1072.45 | 1041.00 | 1036.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 10:15:00 | 1050.55 | 1052.93 | 1043.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 11:00:00 | 1050.55 | 1052.93 | 1043.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1036.90 | 1054.00 | 1049.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1022.85 | 1054.00 | 1049.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 990.45 | 1041.29 | 1043.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 966.00 | 1026.23 | 1036.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 1012.05 | 1006.82 | 1021.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 1012.05 | 1006.82 | 1021.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1012.05 | 1006.82 | 1021.85 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 1052.45 | 1031.75 | 1029.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1070.35 | 1042.17 | 1034.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1169.20 | 1172.10 | 1138.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 1169.55 | 1172.10 | 1138.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 1158.55 | 1167.52 | 1159.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 1158.55 | 1167.52 | 1159.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 1157.40 | 1165.49 | 1159.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 1173.00 | 1165.49 | 1159.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 14:15:00 | 1191.50 | 1199.13 | 1200.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1191.50 | 1199.13 | 1200.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 11:15:00 | 1182.15 | 1193.58 | 1196.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 11:15:00 | 1191.55 | 1182.45 | 1188.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 11:15:00 | 1191.55 | 1182.45 | 1188.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 1191.55 | 1182.45 | 1188.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:30:00 | 1190.00 | 1182.45 | 1188.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 1181.90 | 1182.34 | 1187.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:00:00 | 1164.10 | 1180.51 | 1186.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 1220.50 | 1186.91 | 1187.91 | SL hit (close>static) qty=1.00 sl=1191.85 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 1221.85 | 1193.90 | 1190.99 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 1193.60 | 1196.17 | 1196.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 1189.55 | 1193.81 | 1195.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 1196.15 | 1190.85 | 1192.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 1196.15 | 1190.85 | 1192.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1196.15 | 1190.85 | 1192.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 1196.15 | 1190.85 | 1192.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 1192.00 | 1191.08 | 1192.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 1244.50 | 1191.08 | 1192.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 1230.00 | 1198.86 | 1196.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 09:15:00 | 1272.95 | 1245.78 | 1235.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 1350.00 | 1373.86 | 1346.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 1350.00 | 1373.86 | 1346.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1350.00 | 1373.86 | 1346.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 1350.00 | 1373.86 | 1346.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1334.05 | 1365.90 | 1345.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1337.25 | 1365.90 | 1345.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1340.35 | 1360.79 | 1344.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:15:00 | 1358.85 | 1352.68 | 1343.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-11 10:15:00 | 1494.74 | 1400.70 | 1369.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 13:15:00 | 1400.20 | 1418.50 | 1420.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 14:15:00 | 1394.45 | 1413.69 | 1417.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1341.00 | 1319.67 | 1338.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1341.00 | 1319.67 | 1338.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1341.00 | 1319.67 | 1338.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 1341.00 | 1319.67 | 1338.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1332.15 | 1322.17 | 1337.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 1361.00 | 1322.17 | 1337.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1336.00 | 1324.93 | 1337.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 1338.85 | 1324.93 | 1337.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1358.55 | 1331.66 | 1339.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:15:00 | 1359.00 | 1331.66 | 1339.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1352.25 | 1335.77 | 1340.77 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 1368.45 | 1348.01 | 1345.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 1383.80 | 1358.22 | 1351.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 1327.00 | 1352.74 | 1349.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 1327.00 | 1352.74 | 1349.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1327.00 | 1352.74 | 1349.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1329.25 | 1352.74 | 1349.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1346.80 | 1351.55 | 1349.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 1338.25 | 1351.55 | 1349.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 1319.65 | 1345.17 | 1346.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 1313.60 | 1329.14 | 1335.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 10:15:00 | 1340.95 | 1331.50 | 1336.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 10:15:00 | 1340.95 | 1331.50 | 1336.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 1340.95 | 1331.50 | 1336.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 1340.95 | 1331.50 | 1336.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 1330.65 | 1331.33 | 1335.88 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 1349.50 | 1339.36 | 1338.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 1381.00 | 1349.69 | 1343.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 10:15:00 | 1370.95 | 1373.65 | 1367.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 10:30:00 | 1370.00 | 1373.65 | 1367.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1405.00 | 1388.58 | 1380.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:45:00 | 1373.20 | 1388.58 | 1380.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 1386.55 | 1388.17 | 1381.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:30:00 | 1386.00 | 1388.17 | 1381.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1362.10 | 1382.96 | 1379.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 1362.10 | 1382.96 | 1379.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1364.90 | 1379.35 | 1378.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1349.00 | 1379.35 | 1378.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 1346.80 | 1372.84 | 1375.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1299.85 | 1338.08 | 1354.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1303.50 | 1288.97 | 1315.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1303.50 | 1288.97 | 1315.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1303.50 | 1288.97 | 1315.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:30:00 | 1312.80 | 1288.97 | 1315.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1322.00 | 1297.72 | 1307.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 1325.00 | 1297.72 | 1307.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1328.45 | 1303.86 | 1309.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 1329.90 | 1303.86 | 1309.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 1340.95 | 1316.03 | 1314.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 13:15:00 | 1351.35 | 1323.09 | 1317.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 15:15:00 | 1328.40 | 1329.10 | 1321.49 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:30:00 | 1356.95 | 1336.78 | 1326.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:00:00 | 1357.00 | 1336.78 | 1326.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 15:00:00 | 1359.05 | 1348.32 | 1335.89 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1354.45 | 1351.09 | 1339.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-09 12:15:00 | 1338.60 | 1346.85 | 1340.24 | SL hit (close<ema400) qty=1.00 sl=1340.24 alert=retest1 |

### Cycle 93 — SELL (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 11:15:00 | 1356.95 | 1377.40 | 1377.59 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 1380.45 | 1375.06 | 1374.61 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 1343.40 | 1369.59 | 1372.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 10:15:00 | 1334.00 | 1362.47 | 1369.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1304.40 | 1294.06 | 1316.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1304.40 | 1294.06 | 1316.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1304.40 | 1294.06 | 1316.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 1307.55 | 1294.06 | 1316.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 1310.60 | 1299.47 | 1310.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 15:00:00 | 1310.60 | 1299.47 | 1310.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1307.00 | 1300.97 | 1310.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 1302.30 | 1300.97 | 1310.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1309.50 | 1302.68 | 1310.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:45:00 | 1309.55 | 1302.68 | 1310.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 1304.15 | 1302.97 | 1309.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:30:00 | 1314.15 | 1302.97 | 1309.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 1301.00 | 1302.58 | 1309.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:30:00 | 1308.50 | 1302.58 | 1309.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1306.70 | 1303.98 | 1308.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:30:00 | 1313.50 | 1303.98 | 1308.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1305.00 | 1304.18 | 1307.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 1301.25 | 1304.18 | 1307.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1303.45 | 1304.04 | 1307.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 11:30:00 | 1295.05 | 1302.44 | 1306.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 12:30:00 | 1295.90 | 1301.64 | 1305.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 09:45:00 | 1295.30 | 1301.21 | 1304.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 10:15:00 | 1315.15 | 1304.00 | 1305.02 | SL hit (close>static) qty=1.00 sl=1312.90 alert=retest2 |

### Cycle 96 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 1315.70 | 1306.34 | 1305.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 1333.00 | 1314.11 | 1309.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 1312.65 | 1317.64 | 1312.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 1312.65 | 1317.64 | 1312.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 1312.65 | 1317.64 | 1312.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:45:00 | 1313.10 | 1317.64 | 1312.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1310.95 | 1316.30 | 1312.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:30:00 | 1311.15 | 1316.30 | 1312.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 1310.30 | 1315.10 | 1312.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:45:00 | 1310.00 | 1315.10 | 1312.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1308.05 | 1313.69 | 1311.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 1306.65 | 1313.69 | 1311.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1312.05 | 1313.36 | 1311.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 14:15:00 | 1316.80 | 1313.36 | 1311.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 1326.00 | 1313.92 | 1312.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:45:00 | 1315.55 | 1317.89 | 1315.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 1332.20 | 1317.00 | 1315.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1335.25 | 1320.65 | 1317.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1342.20 | 1326.19 | 1321.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 1303.85 | 1321.82 | 1323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 1303.85 | 1321.82 | 1323.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 1298.45 | 1311.98 | 1317.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 11:15:00 | 1277.45 | 1275.49 | 1287.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 11:45:00 | 1278.70 | 1275.49 | 1287.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 1293.40 | 1279.47 | 1287.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:45:00 | 1296.00 | 1279.47 | 1287.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 1290.15 | 1281.61 | 1287.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:15:00 | 1295.00 | 1281.61 | 1287.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 1295.00 | 1284.29 | 1288.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 1320.30 | 1284.29 | 1288.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 1307.00 | 1292.78 | 1291.48 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 1281.30 | 1289.89 | 1290.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 15:15:00 | 1280.00 | 1286.65 | 1288.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 14:15:00 | 1281.35 | 1278.04 | 1282.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 1281.35 | 1278.04 | 1282.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 1281.35 | 1278.04 | 1282.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 1281.35 | 1278.04 | 1282.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 1274.05 | 1277.24 | 1281.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 1266.80 | 1277.24 | 1281.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:30:00 | 1259.70 | 1255.64 | 1262.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 1203.46 | 1229.24 | 1241.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 1196.71 | 1222.74 | 1237.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 1232.85 | 1216.71 | 1227.91 | SL hit (close>ema200) qty=0.50 sl=1216.71 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 1235.55 | 1230.54 | 1230.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 1241.35 | 1232.70 | 1231.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 10:15:00 | 1225.45 | 1233.15 | 1231.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 10:15:00 | 1225.45 | 1233.15 | 1231.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1225.45 | 1233.15 | 1231.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 1225.45 | 1233.15 | 1231.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 11:15:00 | 1222.65 | 1231.05 | 1231.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 10:15:00 | 1211.10 | 1221.70 | 1225.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 1212.85 | 1209.00 | 1213.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 1212.85 | 1209.00 | 1213.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1212.85 | 1209.00 | 1213.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1192.95 | 1212.90 | 1214.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 1199.45 | 1203.55 | 1209.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 1200.75 | 1201.48 | 1205.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:30:00 | 1201.05 | 1201.03 | 1204.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 1214.90 | 1201.50 | 1202.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:45:00 | 1217.45 | 1201.50 | 1202.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-03 12:15:00 | 1224.95 | 1206.19 | 1204.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 12:15:00 | 1224.95 | 1206.19 | 1204.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 14:15:00 | 1232.45 | 1213.17 | 1208.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1225.25 | 1228.77 | 1220.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 14:00:00 | 1225.25 | 1228.77 | 1220.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1231.40 | 1229.29 | 1221.08 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1187.60 | 1217.18 | 1217.37 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 1239.45 | 1214.22 | 1213.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 12:15:00 | 1251.50 | 1226.01 | 1219.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 1283.75 | 1283.98 | 1267.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:30:00 | 1281.80 | 1283.98 | 1267.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1268.60 | 1277.72 | 1268.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:45:00 | 1268.70 | 1277.72 | 1268.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 1271.25 | 1276.43 | 1268.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 1263.05 | 1276.43 | 1268.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1265.15 | 1274.17 | 1268.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 1266.55 | 1274.17 | 1268.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1267.45 | 1272.83 | 1268.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 1261.90 | 1272.83 | 1268.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 1266.00 | 1271.46 | 1268.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 1262.80 | 1271.46 | 1268.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1271.00 | 1271.39 | 1269.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 1264.00 | 1271.39 | 1269.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1280.40 | 1273.19 | 1270.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:15:00 | 1308.05 | 1278.11 | 1275.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 1280.35 | 1299.69 | 1301.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 1280.35 | 1299.69 | 1301.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 1272.00 | 1294.15 | 1298.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 14:15:00 | 1213.45 | 1213.17 | 1231.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 15:00:00 | 1213.45 | 1213.17 | 1231.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1227.90 | 1214.96 | 1227.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:00:00 | 1227.90 | 1214.96 | 1227.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 1239.00 | 1219.77 | 1228.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:45:00 | 1237.90 | 1219.77 | 1228.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 1232.00 | 1222.22 | 1229.04 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 15:15:00 | 1247.00 | 1234.79 | 1233.71 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1192.70 | 1226.38 | 1229.98 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 15:15:00 | 1245.15 | 1229.31 | 1228.71 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 1222.65 | 1227.98 | 1228.16 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 1250.80 | 1231.51 | 1229.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 1261.65 | 1244.10 | 1238.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 1275.65 | 1283.38 | 1272.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 18:15:00 | 1275.65 | 1283.38 | 1272.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1275.65 | 1283.38 | 1272.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1275.65 | 1283.38 | 1272.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1261.25 | 1278.95 | 1271.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1261.25 | 1278.95 | 1271.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1280.95 | 1279.35 | 1272.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 1298.35 | 1283.89 | 1276.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 1286.15 | 1287.01 | 1279.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:45:00 | 1287.10 | 1285.63 | 1281.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 1258.00 | 1276.55 | 1278.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 11:15:00 | 1258.00 | 1276.55 | 1278.69 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 10:15:00 | 1315.30 | 1282.41 | 1279.16 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 1208.40 | 1278.34 | 1280.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1183.15 | 1206.00 | 1229.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 1101.40 | 1100.78 | 1128.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:00:00 | 1101.40 | 1100.78 | 1128.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1082.80 | 1099.87 | 1117.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 1069.10 | 1087.04 | 1106.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1068.20 | 1084.39 | 1103.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 15:00:00 | 1068.50 | 1081.21 | 1100.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 1068.25 | 1076.39 | 1088.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 1074.80 | 1069.28 | 1077.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 15:00:00 | 1074.80 | 1069.28 | 1077.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 1075.00 | 1070.43 | 1077.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 09:30:00 | 1064.15 | 1070.36 | 1076.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 10:15:00 | 1065.60 | 1070.36 | 1076.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 1086.05 | 1073.50 | 1077.52 | SL hit (close>static) qty=1.00 sl=1078.80 alert=retest2 |

### Cycle 114 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 1086.70 | 1081.05 | 1080.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1106.80 | 1088.37 | 1084.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 15:15:00 | 1095.50 | 1096.55 | 1091.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 15:15:00 | 1095.50 | 1096.55 | 1091.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 1095.50 | 1096.55 | 1091.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 1117.15 | 1096.55 | 1091.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 1100.45 | 1105.14 | 1105.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 14:15:00 | 1100.45 | 1105.14 | 1105.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 1080.00 | 1099.61 | 1103.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 1099.30 | 1084.81 | 1091.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 1099.30 | 1084.81 | 1091.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1099.30 | 1084.81 | 1091.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 1107.00 | 1084.81 | 1091.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1116.30 | 1091.11 | 1093.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:00:00 | 1116.30 | 1091.11 | 1093.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 12:15:00 | 1115.30 | 1098.04 | 1096.41 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 12:15:00 | 1089.80 | 1097.22 | 1097.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 1085.50 | 1094.15 | 1096.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 09:15:00 | 1084.70 | 1082.59 | 1087.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 1084.70 | 1082.59 | 1087.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1084.70 | 1082.59 | 1087.17 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 1107.80 | 1091.19 | 1090.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 11:15:00 | 1117.00 | 1102.34 | 1097.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 14:15:00 | 1105.70 | 1106.11 | 1100.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 14:45:00 | 1105.00 | 1106.11 | 1100.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1089.55 | 1103.58 | 1100.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:15:00 | 1088.20 | 1103.58 | 1100.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1087.75 | 1100.42 | 1099.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 1086.10 | 1100.42 | 1099.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 1086.10 | 1097.55 | 1098.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 13:15:00 | 1084.25 | 1093.14 | 1095.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 1088.80 | 1088.49 | 1092.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 1088.80 | 1088.49 | 1092.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1088.80 | 1088.49 | 1092.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 1073.65 | 1087.18 | 1090.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 1072.35 | 1080.85 | 1084.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:15:00 | 1019.97 | 1041.11 | 1054.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:15:00 | 1018.73 | 1041.11 | 1054.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 1026.00 | 1025.36 | 1035.44 | SL hit (close>ema200) qty=0.50 sl=1025.36 alert=retest2 |

### Cycle 120 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 983.00 | 976.44 | 975.71 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 962.20 | 973.79 | 974.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 11:15:00 | 960.00 | 971.03 | 973.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 963.40 | 957.55 | 962.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 963.40 | 957.55 | 962.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 963.40 | 957.55 | 962.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 963.40 | 957.55 | 962.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 961.25 | 958.29 | 962.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 963.25 | 958.29 | 962.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 964.95 | 959.62 | 962.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 971.55 | 959.62 | 962.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 969.60 | 961.62 | 963.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 973.25 | 961.62 | 963.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 966.90 | 962.67 | 963.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 968.35 | 962.67 | 963.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 969.35 | 965.12 | 964.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 985.30 | 971.03 | 967.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 982.10 | 982.96 | 978.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:15:00 | 969.50 | 982.96 | 978.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 966.55 | 979.68 | 977.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 964.05 | 979.68 | 977.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 959.55 | 975.65 | 975.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 959.55 | 975.65 | 975.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 958.00 | 972.12 | 974.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 948.90 | 964.99 | 970.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 977.55 | 962.38 | 967.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 977.55 | 962.38 | 967.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 977.55 | 962.38 | 967.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 974.35 | 962.38 | 967.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 974.05 | 964.72 | 967.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 12:15:00 | 969.90 | 966.49 | 968.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 981.35 | 971.67 | 970.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 981.35 | 971.67 | 970.51 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 962.30 | 969.76 | 970.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 11:15:00 | 955.80 | 961.74 | 964.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 10:15:00 | 954.75 | 953.53 | 958.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 11:00:00 | 954.75 | 953.53 | 958.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 951.45 | 953.11 | 958.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:00:00 | 942.50 | 950.99 | 956.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 895.38 | 918.04 | 934.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 926.05 | 913.56 | 927.88 | SL hit (close>ema200) qty=0.50 sl=913.56 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 954.15 | 926.87 | 926.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 959.00 | 933.30 | 929.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 963.40 | 964.71 | 950.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 14:15:00 | 952.00 | 958.85 | 952.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 952.00 | 958.85 | 952.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 950.55 | 958.85 | 952.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 953.00 | 957.68 | 952.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 955.90 | 957.68 | 952.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 955.30 | 957.21 | 952.72 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 935.00 | 950.13 | 951.87 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 967.25 | 949.67 | 947.91 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 943.15 | 951.81 | 951.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 938.40 | 949.13 | 950.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 14:15:00 | 911.30 | 903.78 | 915.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 14:15:00 | 911.30 | 903.78 | 915.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 911.30 | 903.78 | 915.53 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 929.50 | 920.99 | 920.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 934.50 | 923.69 | 921.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 981.95 | 982.41 | 967.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 981.95 | 982.41 | 967.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 951.95 | 981.76 | 972.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:30:00 | 950.60 | 981.76 | 972.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 952.55 | 975.92 | 970.48 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 952.15 | 965.77 | 966.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 10:15:00 | 938.00 | 952.78 | 959.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 919.75 | 910.26 | 921.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 10:00:00 | 919.75 | 910.26 | 921.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 914.65 | 911.13 | 920.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 922.15 | 911.13 | 920.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 902.10 | 910.23 | 916.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:00:00 | 894.20 | 906.06 | 910.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 905.00 | 899.14 | 898.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 905.00 | 899.14 | 898.58 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 13:15:00 | 895.10 | 900.39 | 900.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 894.00 | 899.38 | 900.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 898.05 | 897.60 | 899.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 12:15:00 | 898.05 | 897.60 | 899.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 898.05 | 897.60 | 899.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 898.05 | 897.60 | 899.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 900.15 | 898.11 | 899.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 905.15 | 898.11 | 899.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 902.95 | 899.08 | 899.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 902.95 | 899.08 | 899.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 901.00 | 899.46 | 899.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 892.00 | 899.46 | 899.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 900.55 | 899.64 | 899.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:45:00 | 907.55 | 899.64 | 899.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 892.60 | 898.23 | 899.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:30:00 | 890.15 | 897.26 | 898.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 14:00:00 | 892.15 | 896.24 | 897.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:15:00 | 890.10 | 895.67 | 897.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 903.15 | 896.28 | 897.45 | SL hit (close>static) qty=1.00 sl=902.95 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 905.90 | 899.25 | 898.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 910.50 | 903.56 | 901.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 909.80 | 910.27 | 906.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 909.80 | 910.27 | 906.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 909.80 | 910.27 | 906.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 917.30 | 911.66 | 907.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 13:15:00 | 905.45 | 911.27 | 908.35 | SL hit (close<static) qty=1.00 sl=905.50 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 886.25 | 904.61 | 905.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 15:15:00 | 880.80 | 885.55 | 891.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 820.45 | 816.68 | 833.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 820.45 | 816.68 | 833.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 822.65 | 818.56 | 824.72 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 837.80 | 827.79 | 827.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 842.30 | 831.41 | 829.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 889.05 | 890.72 | 874.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 10:00:00 | 889.05 | 890.72 | 874.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 894.85 | 891.47 | 882.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:45:00 | 901.05 | 892.92 | 883.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:45:00 | 903.00 | 894.53 | 885.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:15:00 | 900.20 | 895.52 | 886.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 900.55 | 899.25 | 892.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 896.00 | 898.12 | 893.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:45:00 | 900.95 | 898.36 | 895.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 12:45:00 | 900.40 | 898.35 | 895.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 13:30:00 | 901.00 | 898.04 | 895.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 901.30 | 897.34 | 895.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 894.55 | 896.78 | 895.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:15:00 | 892.10 | 896.78 | 895.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 895.05 | 896.43 | 895.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 11:30:00 | 897.00 | 896.43 | 895.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:45:00 | 897.40 | 896.43 | 895.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 14:30:00 | 897.00 | 895.50 | 895.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 15:15:00 | 889.00 | 894.20 | 894.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 15:15:00 | 889.00 | 894.20 | 894.83 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 900.00 | 895.84 | 895.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 915.95 | 901.11 | 898.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 14:15:00 | 942.45 | 946.65 | 937.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 942.45 | 946.65 | 937.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 939.70 | 945.26 | 937.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 952.65 | 945.26 | 937.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:45:00 | 958.30 | 947.97 | 939.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 13:30:00 | 951.00 | 949.54 | 943.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 925.05 | 938.46 | 940.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 925.05 | 938.46 | 940.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 924.15 | 933.47 | 937.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 922.85 | 922.49 | 927.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 922.85 | 922.49 | 927.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 922.85 | 922.49 | 927.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 925.95 | 922.49 | 927.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 944.60 | 925.74 | 926.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 944.60 | 925.74 | 926.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 946.00 | 929.79 | 928.44 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 923.95 | 929.59 | 929.83 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 930.50 | 928.69 | 928.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 937.85 | 930.52 | 929.36 | Break + close above crossover candle high |

### Cycle 143 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 916.65 | 928.38 | 928.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 10:15:00 | 911.10 | 924.93 | 927.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-09 12:15:00 | 871.35 | 868.10 | 876.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-09 13:00:00 | 871.35 | 868.10 | 876.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 874.05 | 866.96 | 873.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 15:00:00 | 862.95 | 866.34 | 870.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 12:15:00 | 877.00 | 871.99 | 871.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 877.00 | 871.99 | 871.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 881.45 | 875.70 | 873.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 884.30 | 887.75 | 882.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 872.00 | 887.75 | 882.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 866.10 | 883.42 | 881.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 866.10 | 883.42 | 881.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 872.50 | 881.24 | 880.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 874.30 | 881.24 | 880.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 11:15:00 | 872.00 | 879.39 | 879.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 11:15:00 | 872.00 | 879.39 | 879.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 12:15:00 | 870.50 | 877.61 | 878.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-17 13:15:00 | 879.00 | 877.89 | 878.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 13:15:00 | 879.00 | 877.89 | 878.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 879.00 | 877.89 | 878.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:30:00 | 879.40 | 877.89 | 878.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 878.50 | 878.01 | 878.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:45:00 | 879.35 | 878.01 | 878.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 876.55 | 877.72 | 878.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 886.05 | 877.72 | 878.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 899.10 | 882.00 | 880.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 905.85 | 886.77 | 882.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 911.50 | 914.16 | 904.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 911.50 | 914.16 | 904.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 916.40 | 919.70 | 916.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 911.00 | 919.70 | 916.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 887.90 | 913.34 | 913.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 880.90 | 906.85 | 910.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 897.50 | 894.66 | 901.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 897.50 | 894.66 | 901.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 899.65 | 895.66 | 901.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 899.65 | 895.66 | 901.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 900.55 | 896.64 | 901.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 901.90 | 896.64 | 901.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 898.75 | 897.06 | 901.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 904.05 | 897.06 | 901.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 902.20 | 898.09 | 901.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:00:00 | 902.20 | 898.09 | 901.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 907.90 | 900.05 | 901.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 907.90 | 900.05 | 901.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 909.90 | 902.02 | 902.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 912.70 | 902.02 | 902.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 906.95 | 903.01 | 902.96 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 901.70 | 903.01 | 903.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 890.45 | 900.02 | 901.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 14:15:00 | 877.90 | 877.63 | 884.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 15:00:00 | 877.90 | 877.63 | 884.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 889.00 | 879.66 | 884.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 891.00 | 879.66 | 884.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 894.55 | 882.64 | 885.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 894.55 | 882.64 | 885.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 912.90 | 888.69 | 887.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 931.40 | 905.89 | 896.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 905.30 | 909.28 | 899.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 09:45:00 | 904.15 | 909.28 | 899.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 889.75 | 904.56 | 901.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 889.75 | 904.56 | 901.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 884.20 | 900.49 | 899.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 877.55 | 900.49 | 899.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 868.90 | 894.17 | 896.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 866.05 | 877.40 | 883.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 875.00 | 871.91 | 877.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 11:45:00 | 876.95 | 871.91 | 877.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 878.30 | 872.06 | 876.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 878.30 | 872.06 | 876.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 875.00 | 872.65 | 876.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 09:30:00 | 868.80 | 872.11 | 875.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 900.90 | 877.87 | 877.89 | SL hit (close>static) qty=1.00 sl=879.70 alert=retest2 |

### Cycle 152 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 907.55 | 883.80 | 880.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 916.05 | 905.27 | 898.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 12:15:00 | 911.00 | 914.29 | 908.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:45:00 | 911.25 | 914.29 | 908.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 911.00 | 912.98 | 909.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 915.00 | 912.98 | 909.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 924.10 | 942.41 | 944.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 924.10 | 942.41 | 944.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 917.50 | 937.43 | 941.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 12:15:00 | 915.35 | 915.03 | 922.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:30:00 | 916.00 | 915.03 | 922.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 925.30 | 915.95 | 920.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 926.95 | 915.95 | 920.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 928.40 | 918.44 | 921.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 928.40 | 918.44 | 921.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 926.00 | 923.15 | 922.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 929.50 | 924.42 | 923.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 926.05 | 930.00 | 927.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 926.05 | 930.00 | 927.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 926.05 | 930.00 | 927.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 935.55 | 932.57 | 930.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 936.85 | 934.46 | 931.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 926.50 | 935.66 | 936.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 926.50 | 935.66 | 936.30 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 965.55 | 939.17 | 937.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 987.65 | 975.55 | 964.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 991.05 | 995.93 | 988.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 991.05 | 995.93 | 988.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 988.50 | 994.44 | 988.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 989.80 | 994.44 | 988.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 988.10 | 993.18 | 988.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:30:00 | 995.00 | 989.75 | 988.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 15:15:00 | 986.40 | 987.79 | 987.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 986.40 | 987.79 | 987.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 978.75 | 985.98 | 986.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 14:15:00 | 983.30 | 982.25 | 984.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 983.30 | 982.25 | 984.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 983.30 | 982.25 | 984.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 983.30 | 982.25 | 984.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 971.10 | 980.05 | 983.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:15:00 | 969.50 | 980.05 | 983.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 967.00 | 976.57 | 980.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 964.90 | 974.24 | 979.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 968.15 | 970.99 | 976.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 975.00 | 971.79 | 976.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:30:00 | 972.90 | 971.79 | 976.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 976.25 | 972.68 | 976.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:30:00 | 974.10 | 973.67 | 976.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 14:15:00 | 968.35 | 973.67 | 976.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 956.75 | 974.47 | 976.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1004.05 | 980.38 | 978.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 09:15:00 | 1004.05 | 980.38 | 978.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 10:15:00 | 1033.95 | 991.10 | 983.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 13:15:00 | 997.00 | 1004.87 | 992.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 999.10 | 1004.87 | 992.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 989.85 | 1001.87 | 992.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:45:00 | 993.20 | 1001.87 | 992.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 985.00 | 998.49 | 991.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1009.50 | 998.49 | 991.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 979.90 | 997.24 | 995.32 | SL hit (close<static) qty=1.00 sl=984.10 alert=retest2 |

### Cycle 159 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 986.00 | 993.01 | 993.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 975.00 | 989.41 | 991.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 961.55 | 956.85 | 965.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 961.55 | 956.85 | 965.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 966.70 | 958.82 | 965.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 966.70 | 958.82 | 965.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 966.30 | 960.32 | 965.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 967.25 | 960.32 | 965.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 970.30 | 962.31 | 965.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 970.30 | 962.31 | 965.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 959.75 | 961.80 | 965.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 958.00 | 961.80 | 965.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 973.00 | 963.43 | 965.31 | SL hit (close>static) qty=1.00 sl=972.55 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 976.50 | 967.10 | 966.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 991.80 | 972.94 | 969.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 972.00 | 978.82 | 973.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 972.00 | 978.82 | 973.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 972.00 | 978.82 | 973.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 973.05 | 978.82 | 973.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 977.30 | 978.52 | 973.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 983.20 | 975.06 | 973.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 13:45:00 | 981.05 | 984.12 | 979.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 981.55 | 982.63 | 981.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 976.90 | 981.83 | 981.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 976.90 | 981.83 | 981.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 973.80 | 980.22 | 981.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 986.90 | 980.84 | 981.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 986.90 | 980.84 | 981.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 986.90 | 980.84 | 981.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 991.40 | 980.84 | 981.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 10:15:00 | 984.80 | 981.63 | 981.59 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 979.10 | 981.53 | 981.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 13:15:00 | 978.00 | 980.82 | 981.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 981.60 | 980.59 | 981.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 981.60 | 980.59 | 981.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 981.60 | 980.59 | 981.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 981.00 | 980.59 | 981.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 986.35 | 981.74 | 981.49 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 973.55 | 980.94 | 981.61 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 985.30 | 982.12 | 982.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 999.60 | 986.73 | 984.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 13:15:00 | 1002.00 | 1002.43 | 994.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1015.00 | 1001.93 | 995.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1019.55 | 1023.62 | 1019.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 1015.60 | 1021.15 | 1019.38 | SL hit (close<ema400) qty=1.00 sl=1019.38 alert=retest1 |

### Cycle 167 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 1009.10 | 1017.59 | 1018.00 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 1020.70 | 1018.49 | 1018.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 1029.30 | 1021.17 | 1019.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1015.20 | 1020.92 | 1019.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1015.20 | 1020.92 | 1019.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1015.20 | 1020.92 | 1019.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1015.20 | 1020.92 | 1019.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 1011.50 | 1019.04 | 1019.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 1007.95 | 1015.65 | 1017.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1001.20 | 1000.25 | 1006.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 1001.20 | 1000.25 | 1006.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1000.25 | 994.94 | 999.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 987.95 | 997.71 | 999.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:45:00 | 990.10 | 992.52 | 994.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 15:15:00 | 992.85 | 993.78 | 994.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:30:00 | 992.45 | 992.68 | 993.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 991.65 | 991.12 | 992.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 992.65 | 991.12 | 992.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 982.50 | 989.30 | 991.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 991.55 | 990.56 | 990.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 991.55 | 990.56 | 990.53 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 988.60 | 990.34 | 990.45 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 993.90 | 990.97 | 990.72 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 985.15 | 989.81 | 990.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 981.10 | 987.42 | 989.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 964.00 | 962.75 | 972.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 14:30:00 | 961.85 | 962.75 | 972.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 959.00 | 939.65 | 948.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 959.00 | 939.65 | 948.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 957.05 | 943.13 | 948.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 951.40 | 943.13 | 948.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 942.00 | 930.17 | 929.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 942.00 | 930.17 | 929.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 945.45 | 935.36 | 932.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 961.25 | 962.24 | 953.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 961.25 | 962.24 | 953.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 956.00 | 960.97 | 955.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 15:00:00 | 972.00 | 958.03 | 955.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:30:00 | 967.45 | 961.76 | 958.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 971.60 | 975.94 | 976.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 971.60 | 975.94 | 976.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 969.60 | 974.20 | 975.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 940.95 | 937.37 | 944.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 940.95 | 937.37 | 944.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 946.85 | 936.43 | 938.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 946.85 | 936.43 | 938.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 955.90 | 940.32 | 940.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 961.35 | 946.96 | 943.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 958.00 | 959.52 | 952.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 958.35 | 959.52 | 952.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 966.45 | 960.91 | 954.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 969.50 | 960.91 | 954.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 971.50 | 962.23 | 958.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 970.50 | 969.34 | 964.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 13:15:00 | 953.85 | 963.78 | 964.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 953.85 | 963.78 | 964.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 952.50 | 961.53 | 963.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 979.00 | 963.66 | 964.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 979.00 | 963.66 | 964.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 979.00 | 963.66 | 964.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 983.70 | 963.66 | 964.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 978.90 | 966.71 | 965.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 12:15:00 | 989.05 | 973.46 | 968.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 975.00 | 975.46 | 971.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:15:00 | 986.70 | 975.46 | 971.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 986.40 | 987.12 | 984.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:30:00 | 992.20 | 988.83 | 985.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 15:00:00 | 992.75 | 989.84 | 986.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 12:15:00 | 1036.04 | 1017.15 | 1004.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 1052.15 | 1057.48 | 1048.30 | SL hit (close<ema200) qty=0.50 sl=1057.48 alert=retest1 |

### Cycle 179 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 1039.75 | 1049.05 | 1050.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 1024.70 | 1039.79 | 1044.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1002.95 | 1001.24 | 1012.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:30:00 | 1002.85 | 1001.24 | 1012.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1002.40 | 994.33 | 1002.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 1013.05 | 994.33 | 1002.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 993.30 | 994.13 | 1002.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 990.00 | 994.13 | 1002.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 986.15 | 991.48 | 996.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1034.40 | 1000.72 | 998.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1034.40 | 1000.72 | 998.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1043.45 | 1009.27 | 1002.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 1050.85 | 1054.02 | 1041.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:45:00 | 1049.00 | 1054.02 | 1041.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1057.80 | 1051.80 | 1045.72 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1046.75 | 1048.99 | 1049.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1044.85 | 1046.99 | 1048.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 1034.80 | 1031.93 | 1037.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 13:00:00 | 1034.80 | 1031.93 | 1037.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1034.55 | 1032.73 | 1036.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 1038.15 | 1032.73 | 1036.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1038.05 | 1033.68 | 1036.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 1041.15 | 1033.68 | 1036.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1034.65 | 1033.87 | 1036.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 1033.75 | 1033.96 | 1035.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 1040.60 | 1035.29 | 1036.39 | SL hit (close>static) qty=1.00 sl=1038.65 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 1045.05 | 1038.26 | 1037.61 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 1028.85 | 1037.17 | 1037.54 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1036.45 | 1035.52 | 1035.42 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 1032.40 | 1034.80 | 1035.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 12:15:00 | 1029.80 | 1033.22 | 1034.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1032.30 | 1027.88 | 1030.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 1032.30 | 1027.88 | 1030.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1032.30 | 1027.88 | 1030.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 1032.30 | 1027.88 | 1030.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1016.00 | 1025.51 | 1029.59 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 1040.90 | 1030.08 | 1029.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 1062.00 | 1045.99 | 1038.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1053.10 | 1054.35 | 1047.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1053.10 | 1054.35 | 1047.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1053.10 | 1054.35 | 1047.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 1061.75 | 1055.55 | 1048.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 1074.00 | 1083.23 | 1083.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1074.00 | 1083.23 | 1083.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1068.80 | 1080.34 | 1082.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 1059.00 | 1054.76 | 1062.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:00:00 | 1059.00 | 1054.76 | 1062.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1071.90 | 1058.19 | 1063.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 1070.50 | 1058.19 | 1063.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1076.50 | 1061.85 | 1064.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 1076.70 | 1061.85 | 1064.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 1081.50 | 1068.83 | 1067.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 1090.50 | 1076.26 | 1071.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1147.00 | 1148.30 | 1128.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:45:00 | 1142.20 | 1148.30 | 1128.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1127.00 | 1140.53 | 1129.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 1127.30 | 1140.53 | 1129.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1127.10 | 1137.84 | 1129.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 1135.60 | 1137.84 | 1129.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 1132.10 | 1135.80 | 1130.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1133.00 | 1135.36 | 1130.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 1135.70 | 1135.43 | 1131.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1136.50 | 1135.48 | 1131.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 1134.00 | 1135.48 | 1131.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1133.00 | 1134.84 | 1132.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1139.30 | 1134.84 | 1132.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 11:15:00 | 1123.90 | 1132.35 | 1131.70 | SL hit (close<static) qty=1.00 sl=1125.60 alert=retest2 |

### Cycle 189 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 1115.00 | 1128.88 | 1130.18 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 1132.80 | 1129.99 | 1129.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 15:15:00 | 1137.50 | 1131.61 | 1130.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 1126.70 | 1130.62 | 1130.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 1126.70 | 1130.62 | 1130.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1126.70 | 1130.62 | 1130.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 1122.80 | 1130.62 | 1130.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 1122.50 | 1129.00 | 1129.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 1117.70 | 1124.60 | 1127.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 1107.50 | 1106.95 | 1114.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 11:45:00 | 1106.10 | 1106.95 | 1114.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1076.00 | 1072.83 | 1083.30 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 1095.80 | 1086.32 | 1086.25 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 1084.30 | 1085.91 | 1086.07 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 1092.30 | 1086.37 | 1086.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 1094.90 | 1088.67 | 1087.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 1104.00 | 1115.77 | 1108.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 1104.00 | 1115.77 | 1108.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1104.00 | 1115.77 | 1108.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1104.00 | 1115.77 | 1108.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1104.80 | 1113.57 | 1108.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:15:00 | 1098.50 | 1113.57 | 1108.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 1096.00 | 1105.18 | 1105.23 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 1110.90 | 1104.26 | 1104.14 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 1096.50 | 1103.66 | 1104.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 11:15:00 | 1086.40 | 1098.36 | 1101.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1075.30 | 1074.17 | 1083.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 1075.30 | 1074.17 | 1083.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1080.20 | 1076.26 | 1080.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 1082.80 | 1076.26 | 1080.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1081.60 | 1077.33 | 1080.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 1081.60 | 1077.33 | 1080.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 1089.90 | 1079.84 | 1081.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 1089.90 | 1079.84 | 1081.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1085.60 | 1081.00 | 1081.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 1081.00 | 1081.00 | 1081.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 14:15:00 | 1086.10 | 1082.02 | 1081.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 14:15:00 | 1086.10 | 1082.02 | 1081.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 09:15:00 | 1091.00 | 1083.65 | 1082.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1120.00 | 1120.39 | 1112.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 1120.00 | 1120.39 | 1112.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1111.90 | 1118.69 | 1112.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1111.90 | 1118.69 | 1112.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1109.60 | 1116.87 | 1111.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 1107.60 | 1116.87 | 1111.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1107.60 | 1115.02 | 1111.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 1107.60 | 1115.02 | 1111.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1108.50 | 1113.71 | 1111.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 1111.80 | 1110.91 | 1110.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 1112.30 | 1111.16 | 1110.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:30:00 | 1110.60 | 1110.67 | 1110.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 12:30:00 | 1111.90 | 1110.81 | 1110.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1111.40 | 1110.93 | 1110.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:30:00 | 1109.90 | 1110.93 | 1110.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1112.10 | 1111.16 | 1110.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:15:00 | 1106.00 | 1111.16 | 1110.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 1106.00 | 1110.13 | 1110.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 1106.00 | 1110.13 | 1110.24 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 09:15:00 | 1118.10 | 1111.72 | 1110.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 1130.00 | 1116.75 | 1114.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 13:15:00 | 1113.90 | 1119.91 | 1116.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 1113.90 | 1119.91 | 1116.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1113.90 | 1119.91 | 1116.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1113.90 | 1119.91 | 1116.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 14:15:00 | 1094.20 | 1114.77 | 1114.79 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 15:15:00 | 1111.80 | 1108.90 | 1108.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1114.10 | 1109.94 | 1109.04 | Break + close above crossover candle high |

### Cycle 203 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1098.10 | 1107.57 | 1108.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 1097.40 | 1103.26 | 1105.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1108.80 | 1102.53 | 1104.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1108.80 | 1102.53 | 1104.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1108.80 | 1102.53 | 1104.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 1108.20 | 1102.53 | 1104.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1102.40 | 1102.51 | 1104.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 1100.50 | 1103.11 | 1104.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 1100.20 | 1103.26 | 1104.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:45:00 | 1099.10 | 1102.13 | 1103.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:15:00 | 1100.20 | 1102.13 | 1103.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1102.00 | 1102.11 | 1103.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1102.00 | 1102.11 | 1103.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1102.50 | 1102.19 | 1103.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1107.20 | 1102.19 | 1103.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 1121.10 | 1105.97 | 1104.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 1121.10 | 1105.97 | 1104.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 1132.70 | 1114.04 | 1108.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 1118.20 | 1123.10 | 1116.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 1118.20 | 1123.10 | 1116.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1118.20 | 1123.10 | 1116.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:30:00 | 1117.10 | 1123.10 | 1116.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1115.50 | 1121.58 | 1116.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 1115.50 | 1121.58 | 1116.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1118.00 | 1120.86 | 1116.74 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 1110.10 | 1115.17 | 1115.39 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1126.50 | 1116.77 | 1115.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 1134.00 | 1122.25 | 1118.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1109.00 | 1121.77 | 1120.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1109.00 | 1121.77 | 1120.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1109.00 | 1121.77 | 1120.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1099.50 | 1121.77 | 1120.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1108.10 | 1119.03 | 1119.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 1104.20 | 1116.07 | 1117.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1105.90 | 1105.10 | 1110.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:45:00 | 1106.20 | 1105.10 | 1110.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1106.90 | 1105.08 | 1108.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 1106.90 | 1105.08 | 1108.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1112.00 | 1106.51 | 1108.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 1102.80 | 1105.19 | 1107.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 1102.80 | 1095.81 | 1100.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 1101.70 | 1093.10 | 1093.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 1101.60 | 1094.80 | 1093.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 1101.60 | 1094.80 | 1093.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 13:15:00 | 1107.00 | 1098.99 | 1096.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 1105.90 | 1113.38 | 1108.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 10:15:00 | 1105.90 | 1113.38 | 1108.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1105.90 | 1113.38 | 1108.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 1104.50 | 1113.38 | 1108.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 1111.00 | 1112.90 | 1108.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:30:00 | 1117.20 | 1114.38 | 1110.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1103.90 | 1112.46 | 1110.10 | SL hit (close<static) qty=1.00 sl=1106.10 alert=retest2 |

### Cycle 209 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1102.20 | 1108.62 | 1108.66 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 14:15:00 | 1116.30 | 1109.02 | 1108.70 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 1099.70 | 1107.20 | 1108.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 1095.50 | 1104.86 | 1106.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1095.20 | 1093.20 | 1099.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 1095.20 | 1093.20 | 1099.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1095.20 | 1093.20 | 1099.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 1097.10 | 1093.20 | 1099.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 1082.00 | 1090.96 | 1097.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:30:00 | 1093.00 | 1090.96 | 1097.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1099.00 | 1093.39 | 1097.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1099.00 | 1093.39 | 1097.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1092.60 | 1093.23 | 1097.29 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1111.80 | 1101.07 | 1099.86 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1095.60 | 1104.39 | 1105.33 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 1115.50 | 1107.46 | 1106.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 1122.00 | 1110.37 | 1107.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 10:15:00 | 1196.20 | 1196.27 | 1182.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:30:00 | 1196.00 | 1196.27 | 1182.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1170.50 | 1191.12 | 1181.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1170.50 | 1191.12 | 1181.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1190.80 | 1191.05 | 1182.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1169.30 | 1191.05 | 1182.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1174.40 | 1188.41 | 1183.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 1220.40 | 1188.41 | 1183.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-10 09:15:00 | 1342.44 | 1312.00 | 1290.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1300.00 | 1327.49 | 1329.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 12:15:00 | 1289.70 | 1305.65 | 1312.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 09:15:00 | 1305.20 | 1299.81 | 1306.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1305.20 | 1299.81 | 1306.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1305.20 | 1299.81 | 1306.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 1303.50 | 1299.81 | 1306.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1304.80 | 1300.81 | 1306.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1304.80 | 1300.81 | 1306.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1297.80 | 1300.21 | 1305.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:00:00 | 1297.50 | 1299.67 | 1305.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 14:00:00 | 1297.40 | 1299.21 | 1304.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 1294.20 | 1296.92 | 1301.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 12:15:00 | 1290.80 | 1297.54 | 1301.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1306.60 | 1294.36 | 1297.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1309.50 | 1294.36 | 1297.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1308.40 | 1297.17 | 1298.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 1299.00 | 1297.58 | 1298.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1296.10 | 1298.92 | 1298.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 1300.70 | 1299.28 | 1299.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 1300.70 | 1299.28 | 1299.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 11:15:00 | 1303.30 | 1300.08 | 1299.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1341.40 | 1342.79 | 1331.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:45:00 | 1341.40 | 1342.79 | 1331.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1339.40 | 1343.96 | 1336.80 | EMA400 retest candle locked (from upside) |

### Cycle 217 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 1328.60 | 1335.34 | 1336.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 1318.80 | 1332.03 | 1334.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 10:15:00 | 1328.10 | 1326.61 | 1330.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 11:00:00 | 1328.10 | 1326.61 | 1330.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1329.50 | 1327.18 | 1330.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 12:15:00 | 1324.20 | 1327.18 | 1330.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 1344.00 | 1330.74 | 1331.67 | SL hit (close>static) qty=1.00 sl=1338.60 alert=retest2 |

### Cycle 218 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 1354.90 | 1335.57 | 1333.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 1417.50 | 1354.20 | 1342.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 1409.00 | 1410.27 | 1391.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 1381.40 | 1410.27 | 1391.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1386.50 | 1405.52 | 1390.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:30:00 | 1386.10 | 1405.52 | 1390.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 1382.90 | 1400.99 | 1389.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 1391.60 | 1400.99 | 1389.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 1362.50 | 1393.30 | 1387.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:00:00 | 1362.50 | 1393.30 | 1387.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 1335.50 | 1374.82 | 1379.63 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1389.30 | 1378.63 | 1377.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1441.00 | 1395.00 | 1385.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 10:15:00 | 1455.00 | 1460.91 | 1439.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:00:00 | 1455.00 | 1460.91 | 1439.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 1450.00 | 1458.56 | 1446.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 1396.90 | 1458.56 | 1446.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 1380.00 | 1442.85 | 1440.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:00:00 | 1380.00 | 1442.85 | 1440.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 1348.20 | 1423.92 | 1432.33 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1468.70 | 1422.13 | 1417.20 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 1428.10 | 1432.68 | 1432.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1370.70 | 1420.29 | 1427.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1391.00 | 1380.91 | 1398.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1391.00 | 1380.91 | 1398.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1391.00 | 1380.91 | 1398.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 1404.50 | 1380.91 | 1398.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 1390.60 | 1382.85 | 1397.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:30:00 | 1385.40 | 1386.08 | 1396.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 1383.80 | 1386.55 | 1396.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1454.90 | 1402.25 | 1401.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1454.90 | 1402.25 | 1401.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1456.30 | 1413.06 | 1406.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1413.60 | 1418.88 | 1411.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 1413.60 | 1418.88 | 1411.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 1413.60 | 1418.88 | 1411.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:45:00 | 1420.20 | 1418.88 | 1411.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1420.00 | 1419.10 | 1412.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1434.90 | 1419.10 | 1412.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1427.00 | 1420.68 | 1413.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1469.00 | 1418.29 | 1417.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 1444.00 | 1444.55 | 1438.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 1445.40 | 1443.64 | 1438.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1470.70 | 1440.68 | 1438.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1487.40 | 1450.03 | 1443.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-07 11:15:00 | 1430.00 | 1446.28 | 1447.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 1430.00 | 1446.28 | 1447.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 12:15:00 | 1427.90 | 1442.61 | 1445.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1433.80 | 1431.88 | 1438.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1433.80 | 1431.88 | 1438.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1433.80 | 1431.88 | 1438.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:30:00 | 1425.30 | 1423.65 | 1434.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 09:15:00 | 1354.03 | 1373.57 | 1386.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1396.40 | 1358.50 | 1369.53 | SL hit (close>ema200) qty=0.50 sl=1358.50 alert=retest2 |

### Cycle 226 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 1399.10 | 1377.35 | 1376.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 1400.20 | 1385.45 | 1380.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 1382.50 | 1388.23 | 1383.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 1382.50 | 1388.23 | 1383.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 1382.50 | 1388.23 | 1383.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 1382.50 | 1388.23 | 1383.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 1385.30 | 1387.64 | 1383.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 1381.50 | 1387.64 | 1383.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1377.20 | 1385.55 | 1383.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:00:00 | 1377.20 | 1385.55 | 1383.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 1380.80 | 1384.60 | 1383.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 1397.50 | 1384.22 | 1383.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 13:15:00 | 1421.00 | 1434.41 | 1435.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 1421.00 | 1434.41 | 1435.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1412.50 | 1430.03 | 1433.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1416.80 | 1413.18 | 1420.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1416.80 | 1413.18 | 1420.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1416.80 | 1413.18 | 1420.99 | EMA400 retest candle locked (from downside) |

### Cycle 228 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1467.80 | 1430.76 | 1426.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1523.80 | 1471.82 | 1451.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 09:15:00 | 1526.70 | 1556.87 | 1532.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1526.70 | 1556.87 | 1532.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1526.70 | 1556.87 | 1532.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 1526.70 | 1556.87 | 1532.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1546.40 | 1554.77 | 1533.60 | EMA400 retest candle locked (from upside) |

### Cycle 229 — SELL (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 15:15:00 | 1526.00 | 1530.53 | 1531.14 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 1553.00 | 1533.07 | 1531.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 1566.30 | 1539.72 | 1535.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1549.30 | 1551.85 | 1542.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 1549.30 | 1551.85 | 1542.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1549.30 | 1551.85 | 1542.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 1551.00 | 1551.85 | 1542.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1583.10 | 1560.28 | 1551.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1610.00 | 1560.28 | 1551.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 15:15:00 | 1589.10 | 1580.66 | 1566.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-24 12:30:00 | 670.20 | 2023-05-25 10:15:00 | 681.75 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2023-05-25 09:15:00 | 663.50 | 2023-05-25 10:15:00 | 681.75 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2023-05-31 13:15:00 | 689.65 | 2023-06-05 10:15:00 | 677.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2023-05-31 15:15:00 | 686.95 | 2023-06-05 10:15:00 | 677.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-06-01 11:30:00 | 686.10 | 2023-06-05 10:15:00 | 677.70 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-06-01 15:15:00 | 690.50 | 2023-06-05 10:15:00 | 677.70 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2023-06-21 14:30:00 | 729.00 | 2023-06-23 09:15:00 | 692.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-21 14:30:00 | 729.00 | 2023-06-26 09:15:00 | 723.20 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2023-07-05 12:45:00 | 727.25 | 2023-07-06 11:15:00 | 734.90 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-07-07 11:30:00 | 736.60 | 2023-07-14 09:15:00 | 810.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-10 10:45:00 | 735.40 | 2023-07-14 09:15:00 | 808.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-07-27 11:15:00 | 773.00 | 2023-07-31 11:15:00 | 778.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-08-03 09:15:00 | 797.80 | 2023-08-07 12:15:00 | 783.40 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2023-08-10 10:30:00 | 752.60 | 2023-08-11 10:15:00 | 766.55 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2023-08-14 09:15:00 | 741.55 | 2023-08-17 09:15:00 | 786.45 | STOP_HIT | 1.00 | -6.05% |
| SELL | retest2 | 2023-08-16 10:45:00 | 751.95 | 2023-08-17 09:15:00 | 786.45 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2023-08-16 11:15:00 | 751.45 | 2023-08-17 09:15:00 | 786.45 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2023-08-28 14:00:00 | 759.95 | 2023-08-31 15:15:00 | 766.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-08-29 11:30:00 | 756.00 | 2023-08-31 15:15:00 | 766.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2023-08-30 10:15:00 | 759.00 | 2023-08-31 15:15:00 | 766.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-08-30 11:15:00 | 758.50 | 2023-08-31 15:15:00 | 766.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-09-04 10:00:00 | 773.70 | 2023-09-07 11:15:00 | 849.92 | TARGET_HIT | 1.00 | 9.85% |
| BUY | retest2 | 2023-09-04 13:45:00 | 772.65 | 2023-09-07 11:15:00 | 850.14 | TARGET_HIT | 1.00 | 10.03% |
| BUY | retest2 | 2023-09-05 10:00:00 | 772.85 | 2023-09-07 11:15:00 | 849.64 | TARGET_HIT | 1.00 | 9.94% |
| BUY | retest2 | 2023-09-05 11:45:00 | 772.40 | 2023-09-07 11:15:00 | 847.17 | TARGET_HIT | 1.00 | 9.68% |
| BUY | retest2 | 2023-09-06 09:15:00 | 770.15 | 2023-09-07 11:15:00 | 848.76 | TARGET_HIT | 1.00 | 10.21% |
| BUY | retest2 | 2023-09-06 10:30:00 | 771.60 | 2023-09-07 11:15:00 | 847.99 | TARGET_HIT | 1.00 | 9.90% |
| BUY | retest2 | 2023-09-06 11:00:00 | 770.90 | 2023-09-12 09:15:00 | 851.07 | TARGET_HIT | 1.00 | 10.40% |
| BUY | retest2 | 2023-10-11 10:15:00 | 864.50 | 2023-10-18 09:15:00 | 867.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2023-10-12 09:15:00 | 865.80 | 2023-10-18 09:15:00 | 867.00 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2023-10-13 14:15:00 | 864.50 | 2023-10-18 09:15:00 | 867.00 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2023-10-30 09:15:00 | 797.80 | 2023-11-09 09:15:00 | 757.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-30 09:15:00 | 797.80 | 2023-11-10 09:15:00 | 748.10 | STOP_HIT | 0.50 | 6.23% |
| BUY | retest2 | 2023-11-17 09:30:00 | 786.00 | 2023-11-17 14:15:00 | 779.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-11-17 10:00:00 | 785.60 | 2023-11-17 14:15:00 | 779.40 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-11-17 10:45:00 | 786.55 | 2023-11-17 14:15:00 | 779.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-12-07 09:30:00 | 893.70 | 2023-12-08 13:15:00 | 882.65 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-12-07 14:45:00 | 892.00 | 2023-12-08 13:15:00 | 882.65 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-12-08 09:15:00 | 900.00 | 2023-12-08 13:15:00 | 882.65 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2023-12-08 11:15:00 | 891.50 | 2023-12-08 13:15:00 | 882.65 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-12-14 11:00:00 | 857.55 | 2023-12-15 12:15:00 | 876.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2023-12-21 12:30:00 | 994.35 | 2023-12-27 11:15:00 | 987.10 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-01-01 15:15:00 | 970.00 | 2024-01-02 12:15:00 | 986.95 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-01-02 09:45:00 | 965.75 | 2024-01-02 12:15:00 | 986.95 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-01-09 14:30:00 | 966.05 | 2024-01-11 11:15:00 | 974.05 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-01-19 10:45:00 | 957.80 | 2024-01-19 14:15:00 | 967.65 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-02-02 10:30:00 | 975.00 | 2024-02-06 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -4.82% |
| SELL | retest2 | 2024-02-02 12:30:00 | 972.05 | 2024-02-06 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -5.14% |
| SELL | retest2 | 2024-02-05 14:15:00 | 976.95 | 2024-02-06 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2024-02-13 14:15:00 | 943.00 | 2024-02-15 10:15:00 | 965.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-02-14 11:00:00 | 941.55 | 2024-02-15 10:15:00 | 965.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-02-21 10:15:00 | 932.25 | 2024-02-23 09:15:00 | 948.45 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-02-22 14:00:00 | 934.00 | 2024-02-23 09:15:00 | 948.45 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-02-29 11:15:00 | 993.00 | 2024-03-06 12:15:00 | 993.75 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-04-03 09:45:00 | 1016.60 | 2024-04-08 10:15:00 | 1000.05 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-04-12 11:00:00 | 971.25 | 2024-04-15 09:15:00 | 922.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 11:00:00 | 971.25 | 2024-04-16 09:15:00 | 985.35 | STOP_HIT | 0.50 | -1.45% |
| BUY | retest2 | 2024-04-22 09:15:00 | 1021.30 | 2024-05-02 09:15:00 | 1123.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-05-09 15:15:00 | 967.75 | 2024-05-10 15:15:00 | 992.85 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-05-23 11:45:00 | 1024.60 | 2024-05-29 09:15:00 | 1059.00 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1173.00 | 2024-06-19 14:15:00 | 1191.50 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2024-06-21 15:00:00 | 1164.10 | 2024-06-24 09:15:00 | 1220.50 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest2 | 2024-07-10 14:15:00 | 1358.85 | 2024-07-11 10:15:00 | 1494.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-08-08 10:30:00 | 1356.95 | 2024-08-09 12:15:00 | 1338.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest1 | 2024-08-08 11:00:00 | 1357.00 | 2024-08-09 12:15:00 | 1338.60 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2024-08-08 15:00:00 | 1359.05 | 2024-08-09 12:15:00 | 1338.60 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-13 10:00:00 | 1379.70 | 2024-08-20 11:15:00 | 1356.95 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1378.75 | 2024-08-20 11:15:00 | 1356.95 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-08-14 09:45:00 | 1379.70 | 2024-08-20 11:15:00 | 1356.95 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-08-16 12:30:00 | 1381.00 | 2024-08-20 11:15:00 | 1356.95 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-08-29 11:30:00 | 1295.05 | 2024-08-30 10:15:00 | 1315.15 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-08-29 12:30:00 | 1295.90 | 2024-08-30 10:15:00 | 1315.15 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-08-30 09:45:00 | 1295.30 | 2024-08-30 10:15:00 | 1315.15 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-09-02 14:15:00 | 1316.80 | 2024-09-06 10:15:00 | 1303.85 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-09-03 09:15:00 | 1326.00 | 2024-09-06 10:15:00 | 1303.85 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-09-03 12:45:00 | 1315.55 | 2024-09-06 10:15:00 | 1303.85 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-09-04 09:15:00 | 1332.20 | 2024-09-06 10:15:00 | 1303.85 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-09-05 09:15:00 | 1342.20 | 2024-09-06 10:15:00 | 1303.85 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2024-09-13 09:15:00 | 1266.80 | 2024-09-19 10:15:00 | 1203.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 09:30:00 | 1259.70 | 2024-09-19 11:15:00 | 1196.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 09:15:00 | 1266.80 | 2024-09-20 09:15:00 | 1232.85 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2024-09-17 09:30:00 | 1259.70 | 2024-09-20 09:15:00 | 1232.85 | STOP_HIT | 0.50 | 2.13% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1192.95 | 2024-10-03 12:15:00 | 1224.95 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-09-30 12:15:00 | 1199.45 | 2024-10-03 12:15:00 | 1224.95 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-10-01 09:30:00 | 1200.75 | 2024-10-03 12:15:00 | 1224.95 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-10-01 10:30:00 | 1201.05 | 2024-10-03 12:15:00 | 1224.95 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-10-16 10:15:00 | 1308.05 | 2024-10-18 11:15:00 | 1280.35 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-11-04 14:00:00 | 1298.35 | 2024-11-06 11:15:00 | 1258.00 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-11-05 09:45:00 | 1286.15 | 2024-11-06 11:15:00 | 1258.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-11-05 13:45:00 | 1287.10 | 2024-11-06 11:15:00 | 1258.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-11-18 12:45:00 | 1069.10 | 2024-11-22 10:15:00 | 1086.05 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-11-18 14:15:00 | 1068.20 | 2024-11-22 10:15:00 | 1086.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-11-18 15:00:00 | 1068.50 | 2024-11-22 13:15:00 | 1086.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-11-19 14:45:00 | 1068.25 | 2024-11-22 13:15:00 | 1086.70 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-11-22 09:30:00 | 1064.15 | 2024-11-22 13:15:00 | 1086.70 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-11-22 10:15:00 | 1065.60 | 2024-11-22 13:15:00 | 1086.70 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-11-26 09:15:00 | 1117.15 | 2024-11-29 14:15:00 | 1100.45 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-12-12 09:15:00 | 1073.65 | 2024-12-18 11:15:00 | 1019.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 1072.35 | 2024-12-18 11:15:00 | 1018.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 1073.65 | 2024-12-19 14:15:00 | 1026.00 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2024-12-13 09:15:00 | 1072.35 | 2024-12-19 14:15:00 | 1026.00 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2025-01-07 12:15:00 | 969.90 | 2025-01-07 14:15:00 | 981.35 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-01-10 13:00:00 | 942.50 | 2025-01-13 13:15:00 | 895.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 13:00:00 | 942.50 | 2025-01-14 09:15:00 | 926.05 | STOP_HIT | 0.50 | 1.75% |
| SELL | retest2 | 2025-02-11 13:00:00 | 894.20 | 2025-02-13 11:15:00 | 905.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-02-18 12:30:00 | 890.15 | 2025-02-19 09:15:00 | 903.15 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-02-18 14:00:00 | 892.15 | 2025-02-19 09:15:00 | 903.15 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-02-18 15:15:00 | 890.10 | 2025-02-19 09:15:00 | 903.15 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-02-21 11:30:00 | 917.30 | 2025-02-21 13:15:00 | 905.45 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-03-11 10:45:00 | 901.05 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-03-11 11:45:00 | 903.00 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-03-11 13:15:00 | 900.20 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-03-12 10:45:00 | 900.55 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-03-13 11:45:00 | 900.95 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-13 12:45:00 | 900.40 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-03-13 13:30:00 | 901.00 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-17 09:15:00 | 901.30 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-03-17 11:30:00 | 897.00 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-03-17 13:45:00 | 897.40 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-03-17 14:30:00 | 897.00 | 2025-03-17 15:15:00 | 889.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-03-24 09:15:00 | 952.65 | 2025-03-25 12:15:00 | 925.05 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-03-24 09:45:00 | 958.30 | 2025-03-25 12:15:00 | 925.05 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2025-03-24 13:30:00 | 951.00 | 2025-03-25 12:15:00 | 925.05 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-04-11 15:00:00 | 862.95 | 2025-04-15 12:15:00 | 877.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-04-17 11:15:00 | 874.30 | 2025-04-17 11:15:00 | 872.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-05-12 09:30:00 | 868.80 | 2025-05-12 10:15:00 | 900.90 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2025-05-16 09:15:00 | 915.00 | 2025-05-20 13:15:00 | 924.10 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-05-28 09:30:00 | 935.55 | 2025-05-30 10:15:00 | 926.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-05-28 10:45:00 | 936.85 | 2025-05-30 10:15:00 | 926.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-06-09 09:30:00 | 995.00 | 2025-06-09 15:15:00 | 986.40 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-11 10:15:00 | 969.50 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-06-11 13:15:00 | 967.00 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-06-11 14:00:00 | 964.90 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2025-06-12 09:30:00 | 968.15 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-06-12 13:30:00 | 974.10 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-06-12 14:15:00 | 968.35 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-06-13 09:15:00 | 956.75 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest2 | 2025-06-16 09:15:00 | 1009.50 | 2025-06-17 09:15:00 | 979.90 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-06-20 15:15:00 | 958.00 | 2025-06-23 09:15:00 | 973.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-25 09:15:00 | 983.20 | 2025-06-27 13:15:00 | 976.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-25 13:45:00 | 981.05 | 2025-06-27 13:15:00 | 976.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-26 12:45:00 | 981.55 | 2025-06-27 13:15:00 | 976.90 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-07-04 09:15:00 | 1015.00 | 2025-07-08 14:15:00 | 1015.60 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-07-15 15:00:00 | 987.95 | 2025-07-23 10:15:00 | 991.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-17 11:45:00 | 990.10 | 2025-07-23 10:15:00 | 991.55 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-07-17 15:15:00 | 992.85 | 2025-07-23 10:15:00 | 991.55 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-07-18 09:30:00 | 992.45 | 2025-07-23 10:15:00 | 991.55 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-07-30 09:15:00 | 951.40 | 2025-08-07 14:15:00 | 942.00 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-08-14 15:00:00 | 972.00 | 2025-08-22 11:15:00 | 971.60 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-08-18 10:30:00 | 967.45 | 2025-08-22 11:15:00 | 971.60 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-09-03 10:15:00 | 969.50 | 2025-09-08 13:15:00 | 953.85 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-09-04 09:30:00 | 971.50 | 2025-09-08 13:15:00 | 953.85 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-09-05 09:45:00 | 970.50 | 2025-09-08 13:15:00 | 953.85 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2025-09-10 09:15:00 | 986.70 | 2025-09-16 12:15:00 | 1036.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-10 09:15:00 | 986.70 | 2025-09-19 10:15:00 | 1052.15 | STOP_HIT | 0.50 | 6.63% |
| BUY | retest2 | 2025-09-12 12:30:00 | 992.20 | 2025-09-23 11:15:00 | 1039.75 | STOP_HIT | 1.00 | 4.79% |
| BUY | retest2 | 2025-09-12 15:00:00 | 992.75 | 2025-09-23 11:15:00 | 1039.75 | STOP_HIT | 1.00 | 4.73% |
| SELL | retest2 | 2025-09-30 11:15:00 | 990.00 | 2025-10-03 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-10-01 10:45:00 | 986.15 | 2025-10-03 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2025-10-16 11:30:00 | 1033.75 | 2025-10-16 12:15:00 | 1040.60 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-10-29 10:45:00 | 1061.75 | 2025-11-04 10:15:00 | 1074.00 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2025-11-13 14:15:00 | 1135.60 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-11-14 10:15:00 | 1132.10 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1133.00 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-14 12:00:00 | 1135.70 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-11-17 09:15:00 | 1139.30 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-10 14:00:00 | 1081.00 | 2025-12-10 14:15:00 | 1086.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-17 09:15:00 | 1111.80 | 2025-12-17 15:15:00 | 1106.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-12-17 09:45:00 | 1112.30 | 2025-12-17 15:15:00 | 1106.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-17 11:30:00 | 1110.60 | 2025-12-17 15:15:00 | 1106.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-17 12:30:00 | 1111.90 | 2025-12-17 15:15:00 | 1106.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-29 11:30:00 | 1100.50 | 2025-12-31 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-29 15:15:00 | 1100.20 | 2025-12-31 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-12-30 13:45:00 | 1099.10 | 2025-12-31 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-12-30 14:15:00 | 1100.20 | 2025-12-31 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-01-08 11:45:00 | 1102.80 | 2026-01-13 10:15:00 | 1101.60 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-01-09 10:45:00 | 1102.80 | 2026-01-13 10:15:00 | 1101.60 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-01-13 10:15:00 | 1101.70 | 2026-01-13 10:15:00 | 1101.60 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-01-16 14:30:00 | 1117.20 | 2026-01-19 09:15:00 | 1103.90 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-02 09:15:00 | 1220.40 | 2026-02-10 09:15:00 | 1342.44 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-19 13:00:00 | 1297.50 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-02-19 14:00:00 | 1297.40 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-02-20 10:30:00 | 1294.20 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-20 12:15:00 | 1290.80 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-02-23 11:45:00 | 1299.00 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-02-24 09:45:00 | 1296.10 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-03-04 12:15:00 | 1324.20 | 2026-03-04 13:15:00 | 1344.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-03-04 13:30:00 | 1323.70 | 2026-03-04 14:15:00 | 1354.90 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-03-24 12:30:00 | 1385.40 | 2026-03-25 09:15:00 | 1454.90 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2026-03-24 13:45:00 | 1383.80 | 2026-03-25 09:15:00 | 1454.90 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1469.00 | 2026-04-07 11:15:00 | 1430.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-04-02 11:45:00 | 1444.00 | 2026-04-07 11:15:00 | 1430.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-04-02 12:30:00 | 1445.40 | 2026-04-07 11:15:00 | 1430.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-04-06 09:15:00 | 1470.70 | 2026-04-07 11:15:00 | 1430.00 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-04-08 10:30:00 | 1425.30 | 2026-04-13 09:15:00 | 1354.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-08 10:30:00 | 1425.30 | 2026-04-15 09:15:00 | 1396.40 | STOP_HIT | 0.50 | 2.03% |
| BUY | retest2 | 2026-04-17 09:15:00 | 1397.50 | 2026-04-23 13:15:00 | 1421.00 | STOP_HIT | 1.00 | 1.68% |
