# LIC Housing Finance Ltd. (LICHSGFIN)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 581.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 108 |
| ALERT2 | 105 |
| ALERT2_SKIP | 54 |
| ALERT3 | 302 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 121 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 121 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 129 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 34 / 95
- **Target hits / Stop hits / Partials:** 2 / 117 / 10
- **Avg / median % per leg:** 0.19% / -0.64%
- **Sum % (uncompounded):** 24.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 8 | 12.9% | 1 | 61 | 0 | -0.72% | -44.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 62 | 8 | 12.9% | 1 | 61 | 0 | -0.72% | -44.8% |
| SELL (all) | 67 | 26 | 38.8% | 1 | 56 | 10 | 1.04% | 69.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 67 | 26 | 38.8% | 1 | 56 | 10 | 1.04% | 69.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 129 | 34 | 26.4% | 2 | 117 | 10 | 0.19% | 24.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 633.00 | 625.73 | 625.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 636.25 | 627.84 | 626.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 625.65 | 630.38 | 628.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 15:15:00 | 625.65 | 630.38 | 628.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 625.65 | 630.38 | 628.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 634.85 | 630.38 | 628.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 11:30:00 | 634.45 | 632.34 | 630.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 15:15:00 | 648.80 | 651.31 | 651.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 15:15:00 | 648.80 | 651.31 | 651.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 639.85 | 649.02 | 650.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 642.95 | 642.46 | 645.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 642.95 | 642.46 | 645.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 642.95 | 642.46 | 645.59 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 649.75 | 645.97 | 645.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 11:15:00 | 653.40 | 647.46 | 646.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 648.45 | 648.69 | 647.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 14:15:00 | 648.45 | 648.69 | 647.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 648.45 | 648.69 | 647.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 648.45 | 648.69 | 647.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 645.00 | 647.95 | 647.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 641.85 | 647.95 | 647.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 635.05 | 645.37 | 646.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 630.45 | 640.89 | 643.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 633.10 | 632.40 | 636.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 13:45:00 | 633.55 | 632.40 | 636.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 662.15 | 638.65 | 638.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 674.65 | 653.89 | 646.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 641.50 | 660.69 | 652.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 641.50 | 660.69 | 652.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 641.50 | 660.69 | 652.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 622.30 | 660.69 | 652.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 589.45 | 646.44 | 646.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 578.80 | 632.91 | 640.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 606.25 | 604.39 | 621.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:00:00 | 606.25 | 604.39 | 621.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 618.55 | 607.22 | 621.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 618.55 | 607.22 | 621.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 620.30 | 611.41 | 620.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 620.30 | 611.41 | 620.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 616.70 | 612.47 | 620.35 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 652.60 | 628.49 | 625.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 657.05 | 645.48 | 636.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 737.95 | 742.62 | 731.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 740.00 | 742.62 | 731.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 735.20 | 741.14 | 732.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 735.65 | 741.14 | 732.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 736.05 | 740.12 | 732.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:30:00 | 736.05 | 740.12 | 732.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 737.60 | 738.73 | 733.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 741.35 | 737.68 | 733.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 12:45:00 | 739.05 | 738.94 | 735.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 13:30:00 | 739.20 | 738.96 | 735.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 14:00:00 | 739.05 | 738.96 | 735.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 738.45 | 738.97 | 736.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-21 12:15:00 | 729.40 | 735.82 | 735.73 | SL hit (close<static) qty=1.00 sl=733.05 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 13:15:00 | 734.80 | 735.62 | 735.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 727.05 | 732.07 | 733.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 12:15:00 | 735.85 | 731.42 | 732.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 735.85 | 731.42 | 732.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 735.85 | 731.42 | 732.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:45:00 | 736.15 | 731.42 | 732.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 740.95 | 733.32 | 733.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:45:00 | 740.85 | 733.32 | 733.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 14:15:00 | 773.20 | 741.30 | 736.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 785.60 | 761.44 | 748.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 803.05 | 809.33 | 799.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 803.05 | 809.33 | 799.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 803.05 | 809.33 | 799.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 797.60 | 809.33 | 799.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 789.25 | 804.20 | 798.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 789.25 | 804.20 | 798.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 793.45 | 802.05 | 798.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 790.40 | 802.05 | 798.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 793.30 | 799.88 | 797.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 15:00:00 | 793.30 | 799.88 | 797.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 795.90 | 799.09 | 797.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 804.80 | 799.09 | 797.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:15:00 | 797.60 | 799.34 | 798.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 792.65 | 799.51 | 799.05 | SL hit (close<static) qty=1.00 sl=793.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 790.20 | 797.65 | 798.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 782.55 | 791.26 | 793.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 11:15:00 | 774.75 | 773.02 | 779.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 11:45:00 | 777.90 | 773.02 | 779.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 779.50 | 774.25 | 778.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 779.50 | 774.25 | 778.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 779.70 | 775.34 | 778.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 786.60 | 775.34 | 778.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 791.50 | 778.57 | 779.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 791.50 | 778.57 | 779.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 11:15:00 | 785.20 | 780.68 | 780.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 786.65 | 781.87 | 780.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 09:15:00 | 784.70 | 785.61 | 783.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 784.70 | 785.61 | 783.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 784.70 | 785.61 | 783.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 784.70 | 785.61 | 783.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 786.80 | 785.85 | 783.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 783.90 | 785.85 | 783.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 783.90 | 785.46 | 783.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 784.55 | 785.46 | 783.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 782.50 | 784.87 | 783.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 779.55 | 784.87 | 783.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 785.30 | 784.95 | 783.70 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 777.65 | 782.56 | 782.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 776.65 | 781.38 | 782.22 | Break + close below crossover candle low |

### Cycle 13 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 795.05 | 784.11 | 783.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 12:15:00 | 801.60 | 789.27 | 785.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 14:15:00 | 806.35 | 807.74 | 799.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 15:00:00 | 806.35 | 807.74 | 799.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 801.55 | 806.51 | 800.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 795.80 | 806.51 | 800.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 790.80 | 803.37 | 799.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 790.80 | 803.37 | 799.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 797.90 | 802.28 | 799.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 791.55 | 802.28 | 799.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 794.85 | 799.53 | 798.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:30:00 | 793.05 | 799.53 | 798.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 15:15:00 | 795.95 | 798.27 | 798.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 778.05 | 794.22 | 796.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 786.40 | 782.72 | 788.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 786.40 | 782.72 | 788.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 793.40 | 784.85 | 788.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 793.90 | 784.85 | 788.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 797.95 | 787.47 | 789.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 797.95 | 787.47 | 789.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 794.20 | 790.94 | 790.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 15:15:00 | 797.00 | 792.15 | 791.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 791.15 | 791.95 | 791.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 791.15 | 791.95 | 791.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 791.15 | 791.95 | 791.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 789.05 | 791.95 | 791.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 797.10 | 792.98 | 791.97 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 781.85 | 790.76 | 791.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 779.15 | 788.43 | 789.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 780.40 | 780.05 | 784.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 10:00:00 | 780.40 | 780.05 | 784.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 787.25 | 781.49 | 785.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 787.25 | 781.49 | 785.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 787.20 | 782.63 | 785.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 769.05 | 784.27 | 785.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 12:15:00 | 793.45 | 778.32 | 777.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 793.45 | 778.32 | 777.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 805.00 | 788.88 | 783.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 791.20 | 792.67 | 787.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 791.20 | 792.67 | 787.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 781.65 | 790.17 | 787.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 781.65 | 790.17 | 787.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 780.00 | 788.13 | 786.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:15:00 | 780.40 | 788.13 | 786.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 776.70 | 785.85 | 785.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 09:15:00 | 772.35 | 777.84 | 781.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 775.75 | 771.68 | 775.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 775.75 | 771.68 | 775.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 775.75 | 771.68 | 775.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:45:00 | 773.55 | 771.68 | 775.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 771.90 | 771.72 | 775.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 11:15:00 | 768.40 | 771.72 | 775.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-05 09:15:00 | 691.56 | 742.86 | 755.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 651.55 | 644.77 | 644.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 658.55 | 649.00 | 646.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 12:15:00 | 662.95 | 665.18 | 659.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 12:15:00 | 662.95 | 665.18 | 659.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 662.95 | 665.18 | 659.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 660.20 | 665.18 | 659.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 671.55 | 679.03 | 675.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 671.55 | 679.03 | 675.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 673.45 | 677.91 | 675.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:15:00 | 674.25 | 677.91 | 675.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:45:00 | 674.00 | 676.30 | 675.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:30:00 | 674.25 | 676.08 | 675.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 664.60 | 672.59 | 673.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 664.60 | 672.59 | 673.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 13:15:00 | 661.55 | 666.87 | 670.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 669.65 | 667.18 | 669.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 669.65 | 667.18 | 669.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 669.65 | 667.18 | 669.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 669.65 | 667.18 | 669.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 670.60 | 667.86 | 669.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:00:00 | 670.60 | 667.86 | 669.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 673.05 | 668.90 | 669.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:30:00 | 675.90 | 668.90 | 669.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 672.85 | 670.33 | 670.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:00:00 | 672.85 | 670.33 | 670.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 685.00 | 673.26 | 671.79 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 669.95 | 674.72 | 674.86 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 677.55 | 675.33 | 675.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 11:15:00 | 679.70 | 676.80 | 675.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 674.30 | 676.92 | 676.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 674.30 | 676.92 | 676.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 674.30 | 676.92 | 676.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 674.30 | 676.92 | 676.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 674.65 | 676.46 | 676.10 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 672.25 | 675.62 | 675.75 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 684.55 | 677.27 | 676.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 14:15:00 | 693.65 | 683.42 | 679.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 683.55 | 686.14 | 682.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 11:15:00 | 683.55 | 686.14 | 682.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 683.55 | 686.14 | 682.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:30:00 | 682.45 | 686.14 | 682.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 691.95 | 687.30 | 683.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 706.45 | 689.44 | 685.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 15:15:00 | 696.75 | 703.07 | 699.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:45:00 | 696.45 | 700.89 | 699.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 13:15:00 | 699.20 | 698.89 | 698.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 13:15:00 | 697.30 | 698.57 | 698.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 697.30 | 698.57 | 698.69 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 699.90 | 698.84 | 698.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 15:15:00 | 700.75 | 699.22 | 698.98 | Break + close above crossover candle high |

### Cycle 28 — SELL (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 09:15:00 | 696.05 | 698.59 | 698.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 10:15:00 | 690.55 | 696.98 | 697.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 10:15:00 | 691.00 | 690.12 | 693.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 691.00 | 690.12 | 693.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 691.00 | 690.12 | 693.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:30:00 | 693.35 | 690.12 | 693.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 688.05 | 689.37 | 692.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:45:00 | 689.00 | 689.37 | 692.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 696.55 | 686.05 | 689.32 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 704.40 | 692.66 | 691.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 718.00 | 702.32 | 697.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 706.55 | 715.36 | 708.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 706.55 | 715.36 | 708.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 706.55 | 715.36 | 708.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 706.55 | 715.36 | 708.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 683.65 | 709.02 | 705.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 683.65 | 709.02 | 705.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 683.55 | 703.92 | 703.92 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 12:15:00 | 681.70 | 699.48 | 701.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 678.70 | 688.26 | 695.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 10:15:00 | 680.10 | 679.17 | 685.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 10:45:00 | 680.10 | 679.17 | 685.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 678.00 | 679.27 | 684.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:45:00 | 681.85 | 679.27 | 684.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 671.35 | 675.91 | 681.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 665.50 | 675.91 | 681.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 10:45:00 | 669.60 | 668.43 | 669.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 11:15:00 | 677.60 | 670.26 | 670.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 677.60 | 670.26 | 670.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 681.00 | 672.41 | 671.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 670.50 | 681.69 | 679.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 670.50 | 681.69 | 679.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 670.50 | 681.69 | 679.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 670.50 | 681.69 | 679.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 667.35 | 678.82 | 678.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 666.80 | 678.82 | 678.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 664.00 | 675.86 | 676.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 12:15:00 | 661.15 | 672.92 | 675.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 666.40 | 665.04 | 668.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 666.40 | 665.04 | 668.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 668.15 | 665.98 | 668.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:15:00 | 669.45 | 665.98 | 668.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 667.00 | 666.18 | 668.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:45:00 | 665.00 | 666.20 | 668.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:00:00 | 665.60 | 666.08 | 667.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:15:00 | 665.30 | 666.37 | 667.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:00:00 | 664.50 | 665.99 | 666.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 662.45 | 664.40 | 665.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 665.50 | 664.40 | 665.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 15:15:00 | 631.75 | 644.24 | 652.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 15:15:00 | 632.32 | 644.24 | 652.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 15:15:00 | 632.03 | 644.24 | 652.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 15:15:00 | 631.27 | 644.24 | 652.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 616.70 | 613.64 | 624.30 | SL hit (close>ema200) qty=0.50 sl=613.64 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 622.85 | 618.39 | 618.35 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 615.55 | 621.38 | 622.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 611.05 | 618.42 | 620.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 613.40 | 613.22 | 616.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 613.40 | 613.22 | 616.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 614.10 | 613.40 | 616.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:45:00 | 615.90 | 613.40 | 616.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 617.85 | 614.29 | 616.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 617.85 | 614.29 | 616.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 619.10 | 615.25 | 616.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 618.40 | 615.25 | 616.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 619.40 | 616.08 | 617.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 616.10 | 616.08 | 617.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 608.85 | 608.67 | 611.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:45:00 | 612.55 | 608.67 | 611.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 598.75 | 596.60 | 602.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 598.75 | 596.60 | 602.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 601.00 | 597.48 | 602.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 601.00 | 597.48 | 602.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 605.15 | 599.01 | 602.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 609.05 | 599.01 | 602.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 600.95 | 599.40 | 602.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 15:15:00 | 598.85 | 599.40 | 602.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 606.30 | 600.69 | 602.34 | SL hit (close>static) qty=1.00 sl=606.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 613.10 | 602.50 | 601.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 12:15:00 | 618.00 | 607.86 | 604.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 10:15:00 | 614.85 | 615.04 | 609.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 11:00:00 | 614.85 | 615.04 | 609.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 630.50 | 634.58 | 628.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 630.15 | 634.58 | 628.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 630.25 | 633.71 | 628.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:30:00 | 627.75 | 633.71 | 628.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 628.15 | 632.60 | 628.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:30:00 | 626.55 | 632.60 | 628.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 625.75 | 631.23 | 628.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 624.95 | 631.23 | 628.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 635.75 | 631.47 | 629.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 635.75 | 631.47 | 629.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 622.75 | 630.13 | 629.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 622.75 | 630.13 | 629.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 614.95 | 627.09 | 627.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 612.50 | 619.78 | 623.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 633.95 | 621.62 | 623.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 633.95 | 621.62 | 623.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 633.95 | 621.62 | 623.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 633.95 | 621.62 | 623.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 631.60 | 623.62 | 624.02 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 630.50 | 624.99 | 624.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 633.85 | 626.77 | 625.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 634.50 | 637.20 | 634.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 10:15:00 | 634.50 | 637.20 | 634.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 634.50 | 637.20 | 634.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 634.50 | 637.20 | 634.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 636.65 | 637.09 | 634.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:45:00 | 634.65 | 637.09 | 634.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 636.25 | 636.73 | 634.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 638.10 | 636.73 | 634.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 632.90 | 635.96 | 634.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 624.20 | 635.96 | 634.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 624.50 | 633.67 | 633.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 619.30 | 627.40 | 629.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 615.40 | 612.70 | 617.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 615.40 | 612.70 | 617.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 606.70 | 611.50 | 616.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 613.50 | 611.50 | 616.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 610.20 | 609.78 | 613.56 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 624.30 | 614.78 | 614.47 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 603.00 | 614.10 | 614.92 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 618.55 | 614.14 | 613.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 631.00 | 618.13 | 615.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 624.65 | 625.60 | 621.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 11:15:00 | 624.95 | 625.60 | 621.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 621.15 | 624.13 | 622.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:45:00 | 621.20 | 624.13 | 622.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 622.00 | 623.70 | 622.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 620.10 | 623.70 | 622.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 625.10 | 633.55 | 632.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 625.10 | 633.55 | 632.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 625.40 | 631.92 | 631.40 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 627.30 | 630.44 | 630.78 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 639.80 | 630.89 | 630.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 10:15:00 | 646.70 | 639.72 | 637.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 14:15:00 | 641.85 | 641.90 | 639.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 14:15:00 | 641.85 | 641.90 | 639.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 641.85 | 641.90 | 639.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:30:00 | 640.20 | 641.90 | 639.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 634.35 | 640.17 | 639.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 634.35 | 640.17 | 639.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 632.05 | 638.54 | 638.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 630.40 | 634.66 | 636.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 630.45 | 629.84 | 632.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 15:00:00 | 630.45 | 629.84 | 632.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 635.95 | 631.01 | 632.79 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 637.45 | 633.92 | 633.75 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 631.00 | 633.71 | 633.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 614.20 | 629.81 | 632.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 624.10 | 623.51 | 627.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:30:00 | 624.50 | 623.51 | 627.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 629.95 | 625.04 | 627.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 631.25 | 625.04 | 627.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 624.10 | 624.85 | 627.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:15:00 | 624.05 | 624.85 | 627.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 621.05 | 624.91 | 626.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 592.85 | 599.01 | 607.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 590.00 | 599.01 | 607.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 595.50 | 593.88 | 599.85 | SL hit (close>ema200) qty=0.50 sl=593.88 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 597.60 | 588.60 | 588.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 10:15:00 | 600.60 | 591.00 | 589.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 12:15:00 | 594.05 | 596.54 | 594.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 12:15:00 | 594.05 | 596.54 | 594.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 594.05 | 596.54 | 594.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 594.05 | 596.54 | 594.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 588.65 | 594.97 | 593.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 588.65 | 594.97 | 593.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 590.60 | 594.09 | 593.26 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 587.50 | 592.41 | 592.62 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 596.65 | 593.03 | 592.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 14:15:00 | 598.75 | 594.17 | 593.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 599.40 | 599.66 | 597.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 10:15:00 | 599.40 | 599.66 | 597.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 599.40 | 599.66 | 597.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 597.95 | 599.66 | 597.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 604.60 | 600.58 | 598.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:45:00 | 606.85 | 601.71 | 599.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 611.65 | 603.17 | 600.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:30:00 | 606.15 | 608.85 | 605.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 10:00:00 | 606.25 | 608.85 | 605.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 597.55 | 606.59 | 604.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 597.55 | 606.59 | 604.97 | SL hit (close<static) qty=1.00 sl=597.80 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 598.30 | 603.44 | 603.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 591.25 | 601.00 | 602.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 588.85 | 588.27 | 592.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 10:15:00 | 586.60 | 588.27 | 592.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 551.75 | 547.79 | 553.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 551.75 | 547.79 | 553.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 552.05 | 548.64 | 553.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:15:00 | 554.05 | 548.64 | 553.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 554.05 | 549.72 | 553.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 553.60 | 549.72 | 553.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 556.80 | 551.14 | 553.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 556.80 | 551.14 | 553.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 557.00 | 552.31 | 553.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 556.60 | 552.31 | 553.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 552.35 | 553.33 | 553.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 15:00:00 | 552.35 | 553.33 | 553.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 553.90 | 553.44 | 553.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 567.95 | 553.44 | 553.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 568.90 | 556.53 | 555.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 573.20 | 566.64 | 563.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 566.65 | 568.04 | 565.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 566.65 | 568.04 | 565.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 561.50 | 566.73 | 565.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 561.50 | 566.73 | 565.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 565.85 | 566.55 | 565.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 563.85 | 566.55 | 565.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 564.75 | 566.19 | 565.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 563.40 | 566.19 | 565.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 564.10 | 565.77 | 564.98 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 561.95 | 564.10 | 564.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 555.85 | 562.45 | 563.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 555.95 | 555.15 | 558.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 555.95 | 555.15 | 558.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 561.15 | 556.34 | 558.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 561.15 | 556.34 | 558.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 562.25 | 557.52 | 559.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 564.45 | 557.52 | 559.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 578.50 | 562.65 | 561.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 13:15:00 | 581.00 | 566.32 | 563.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 573.70 | 575.04 | 570.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 573.70 | 575.04 | 570.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 572.60 | 574.55 | 570.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 571.40 | 574.55 | 570.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 566.65 | 572.77 | 570.29 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 14:15:00 | 564.80 | 568.49 | 568.90 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 575.40 | 568.73 | 568.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 578.25 | 570.64 | 569.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-28 14:15:00 | 571.40 | 571.75 | 570.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 14:15:00 | 571.40 | 571.75 | 570.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 571.40 | 571.75 | 570.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 15:00:00 | 571.40 | 571.75 | 570.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 571.35 | 571.67 | 570.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 579.05 | 571.67 | 570.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 10:15:00 | 572.40 | 574.15 | 573.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 11:45:00 | 574.40 | 573.92 | 573.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 565.00 | 582.56 | 583.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 565.00 | 582.56 | 583.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 15:15:00 | 557.95 | 566.86 | 574.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 569.10 | 567.31 | 573.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 569.10 | 567.31 | 573.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 569.10 | 567.31 | 573.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 574.00 | 567.31 | 573.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 574.80 | 568.81 | 573.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 574.80 | 568.81 | 573.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 573.55 | 569.75 | 573.84 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 584.90 | 575.86 | 575.53 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 15:15:00 | 573.30 | 575.84 | 575.89 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 576.55 | 575.99 | 575.95 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 573.95 | 575.73 | 575.85 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 12:15:00 | 577.30 | 576.05 | 575.99 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 574.50 | 575.74 | 575.85 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 09:15:00 | 578.40 | 576.32 | 576.09 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 566.60 | 575.71 | 576.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 564.20 | 572.16 | 574.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 542.90 | 542.05 | 551.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 540.35 | 542.05 | 551.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 552.60 | 544.07 | 549.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 552.60 | 544.07 | 549.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 552.90 | 545.84 | 549.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 550.30 | 547.90 | 549.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 550.40 | 548.52 | 549.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 522.78 | 535.00 | 541.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 522.88 | 535.00 | 541.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 535.40 | 534.11 | 538.60 | SL hit (close>ema200) qty=0.50 sl=534.11 alert=retest2 |

### Cycle 65 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 541.80 | 535.27 | 534.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 542.40 | 537.55 | 536.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 539.35 | 542.87 | 540.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 539.35 | 542.87 | 540.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 539.35 | 542.87 | 540.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 539.35 | 542.87 | 540.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 540.85 | 542.47 | 540.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 544.90 | 542.63 | 541.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 10:15:00 | 537.40 | 541.03 | 540.67 | SL hit (close<static) qty=1.00 sl=538.05 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 538.25 | 540.18 | 540.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 532.80 | 537.71 | 539.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 498.10 | 495.95 | 505.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 498.10 | 495.95 | 505.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 500.45 | 498.45 | 502.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:30:00 | 501.55 | 498.45 | 502.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 505.10 | 499.78 | 502.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 505.10 | 499.78 | 502.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 504.60 | 500.74 | 502.64 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 519.05 | 506.67 | 505.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 520.85 | 511.59 | 507.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 528.55 | 529.24 | 523.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 528.55 | 529.24 | 523.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 528.70 | 531.07 | 528.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 526.40 | 531.07 | 528.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 526.85 | 530.22 | 528.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 521.30 | 530.22 | 528.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 526.50 | 529.48 | 528.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:45:00 | 527.60 | 529.40 | 528.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:30:00 | 527.65 | 529.13 | 528.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 12:30:00 | 528.90 | 529.10 | 528.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 11:15:00 | 520.95 | 528.58 | 528.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 520.95 | 528.58 | 528.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 520.00 | 523.06 | 525.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 524.45 | 520.29 | 522.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 524.45 | 520.29 | 522.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 524.45 | 520.29 | 522.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 535.45 | 520.29 | 522.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 522.70 | 520.77 | 522.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:45:00 | 521.00 | 520.82 | 522.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 520.75 | 520.98 | 522.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 533.25 | 523.39 | 523.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 533.25 | 523.39 | 523.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 535.45 | 527.31 | 525.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 573.80 | 575.29 | 568.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:15:00 | 573.50 | 575.29 | 568.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 570.35 | 573.59 | 568.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 567.00 | 573.59 | 568.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 569.45 | 572.76 | 568.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 569.65 | 572.76 | 568.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 573.25 | 572.86 | 569.13 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 561.70 | 567.77 | 568.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 558.85 | 563.70 | 565.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 12:15:00 | 563.30 | 563.09 | 564.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-01 12:45:00 | 564.10 | 563.09 | 564.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 555.85 | 560.92 | 563.13 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 573.75 | 565.15 | 564.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 576.80 | 567.48 | 565.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 565.05 | 570.50 | 568.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 565.05 | 570.50 | 568.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 565.05 | 570.50 | 568.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 565.05 | 570.50 | 568.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 569.35 | 570.27 | 568.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 566.20 | 570.27 | 568.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 563.55 | 568.93 | 567.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 563.55 | 568.93 | 567.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 563.10 | 567.76 | 567.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:15:00 | 563.60 | 567.76 | 567.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 560.70 | 566.35 | 566.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 529.75 | 557.61 | 562.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 548.80 | 545.89 | 553.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 562.95 | 545.89 | 553.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 558.45 | 548.40 | 553.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 563.80 | 548.40 | 553.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 567.80 | 556.44 | 556.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 13:15:00 | 573.40 | 559.83 | 557.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 549.65 | 559.51 | 558.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 549.65 | 559.51 | 558.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 549.65 | 559.51 | 558.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:00:00 | 549.65 | 559.51 | 558.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 10:15:00 | 546.90 | 556.99 | 557.36 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 560.90 | 557.77 | 557.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 563.00 | 558.82 | 558.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 13:15:00 | 556.50 | 558.35 | 558.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 13:15:00 | 556.50 | 558.35 | 558.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 556.50 | 558.35 | 558.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 13:45:00 | 557.95 | 558.35 | 558.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 556.55 | 557.99 | 557.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 15:00:00 | 556.55 | 557.99 | 557.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 557.10 | 557.81 | 557.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 565.00 | 557.81 | 557.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-24 09:15:00 | 621.50 | 611.28 | 608.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 591.90 | 607.18 | 608.44 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 609.70 | 605.06 | 604.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 620.00 | 608.52 | 606.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 610.05 | 611.07 | 608.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 14:00:00 | 610.05 | 611.07 | 608.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 608.00 | 610.46 | 608.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 608.00 | 610.46 | 608.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 605.20 | 609.41 | 608.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 601.00 | 609.41 | 608.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 607.65 | 608.67 | 608.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:30:00 | 608.10 | 608.67 | 608.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 608.95 | 608.73 | 608.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:30:00 | 608.40 | 608.73 | 608.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 606.10 | 608.20 | 607.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:30:00 | 607.75 | 608.20 | 607.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 602.10 | 606.98 | 607.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 598.00 | 604.33 | 606.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 611.80 | 605.83 | 606.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 611.80 | 605.83 | 606.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 611.80 | 605.83 | 606.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 611.80 | 605.83 | 606.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 609.70 | 606.60 | 606.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 606.95 | 606.60 | 606.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 609.85 | 605.97 | 605.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 609.85 | 605.97 | 605.74 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 600.05 | 605.59 | 605.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 10:15:00 | 595.65 | 603.60 | 604.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 592.75 | 591.80 | 597.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 592.75 | 591.80 | 597.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 597.00 | 593.42 | 596.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 597.85 | 593.42 | 596.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 597.10 | 594.15 | 596.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:15:00 | 598.95 | 594.15 | 596.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 599.95 | 595.31 | 597.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:30:00 | 599.75 | 595.31 | 597.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 599.40 | 596.13 | 597.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 600.55 | 596.13 | 597.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 594.85 | 596.22 | 597.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 591.85 | 595.19 | 596.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:00:00 | 591.15 | 579.09 | 583.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 12:15:00 | 595.00 | 587.30 | 586.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 595.00 | 587.30 | 586.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 602.60 | 590.36 | 587.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 603.05 | 604.89 | 598.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 603.05 | 604.89 | 598.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 613.05 | 605.74 | 599.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:00:00 | 618.60 | 610.02 | 604.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:45:00 | 617.00 | 617.57 | 610.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:15:00 | 617.40 | 617.57 | 610.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:30:00 | 618.60 | 619.77 | 614.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 617.90 | 618.87 | 615.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:30:00 | 615.35 | 618.87 | 615.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 606.40 | 617.03 | 615.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 601.70 | 617.03 | 615.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 606.40 | 614.90 | 614.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 606.10 | 614.90 | 614.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-19 11:15:00 | 604.85 | 612.89 | 613.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 604.85 | 612.89 | 613.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 13:15:00 | 602.20 | 609.43 | 612.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 596.30 | 593.11 | 597.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 596.30 | 593.11 | 597.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 596.30 | 593.11 | 597.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 596.30 | 593.11 | 597.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 593.75 | 590.48 | 593.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 593.75 | 590.48 | 593.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 593.80 | 591.15 | 593.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 594.35 | 591.15 | 593.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 593.35 | 591.59 | 593.80 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 598.30 | 595.32 | 595.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 600.50 | 596.76 | 595.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 600.45 | 600.91 | 598.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 600.45 | 600.91 | 598.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 600.45 | 600.91 | 598.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 600.45 | 600.91 | 598.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 600.95 | 601.68 | 600.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:15:00 | 600.90 | 601.68 | 600.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 600.90 | 601.53 | 600.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 602.15 | 601.53 | 600.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 598.00 | 600.58 | 599.99 | SL hit (close<static) qty=1.00 sl=599.75 alert=retest2 |

### Cycle 84 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 594.15 | 599.29 | 599.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 592.90 | 598.01 | 598.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 596.70 | 595.68 | 597.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 11:15:00 | 596.70 | 595.68 | 597.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 596.70 | 595.68 | 597.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 596.70 | 595.68 | 597.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 595.75 | 595.70 | 596.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 597.05 | 595.70 | 596.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 595.80 | 595.12 | 596.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 595.80 | 595.12 | 596.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 595.15 | 595.12 | 596.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 595.25 | 595.12 | 596.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 589.90 | 594.08 | 595.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:30:00 | 588.60 | 593.04 | 595.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 588.05 | 591.91 | 594.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 598.95 | 594.72 | 594.87 | SL hit (close>static) qty=1.00 sl=596.95 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 598.30 | 595.44 | 595.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 600.25 | 596.79 | 595.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 602.15 | 604.66 | 601.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 602.15 | 604.66 | 601.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 602.15 | 604.66 | 601.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 601.75 | 604.66 | 601.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 603.65 | 604.45 | 602.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:00:00 | 606.85 | 604.46 | 603.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 599.55 | 602.47 | 602.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 599.55 | 602.47 | 602.62 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 608.45 | 603.46 | 603.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 614.80 | 606.62 | 604.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 619.00 | 619.41 | 615.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:45:00 | 617.95 | 619.41 | 615.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 623.50 | 619.39 | 616.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 626.10 | 620.24 | 617.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 11:15:00 | 625.25 | 620.24 | 617.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 612.35 | 618.83 | 618.54 | SL hit (close<static) qty=1.00 sl=615.50 alert=retest2 |

### Cycle 88 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 613.65 | 617.79 | 618.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 606.35 | 615.50 | 617.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 601.25 | 600.17 | 604.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 601.25 | 600.17 | 604.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 602.05 | 600.55 | 604.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 602.65 | 600.55 | 604.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 605.90 | 602.08 | 604.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 605.90 | 602.08 | 604.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 607.20 | 603.11 | 604.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 606.75 | 603.11 | 604.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 607.40 | 604.72 | 605.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 603.85 | 603.86 | 604.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 603.50 | 603.22 | 604.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 602.60 | 603.53 | 604.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 604.75 | 593.55 | 593.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 604.75 | 593.55 | 593.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 608.20 | 596.48 | 594.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 601.60 | 601.71 | 598.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 601.60 | 601.71 | 598.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 611.15 | 614.98 | 611.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 611.15 | 614.98 | 611.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 611.25 | 614.23 | 611.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:45:00 | 612.50 | 613.98 | 612.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 619.30 | 613.57 | 612.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 612.75 | 616.22 | 614.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 609.40 | 613.91 | 613.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 609.40 | 613.91 | 613.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 607.35 | 609.65 | 610.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 605.60 | 605.03 | 606.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:45:00 | 605.70 | 605.03 | 606.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 602.70 | 604.57 | 606.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:45:00 | 600.85 | 604.37 | 605.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 610.25 | 605.57 | 605.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 610.25 | 605.57 | 605.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 612.80 | 607.49 | 606.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 631.00 | 636.02 | 631.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 10:15:00 | 631.00 | 636.02 | 631.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 631.00 | 636.02 | 631.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 631.00 | 636.02 | 631.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 632.40 | 635.29 | 631.21 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 624.00 | 628.78 | 629.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 13:15:00 | 621.75 | 624.69 | 626.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 619.30 | 617.84 | 621.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 619.30 | 617.84 | 621.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 621.30 | 618.53 | 621.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 619.60 | 618.53 | 621.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 617.75 | 618.38 | 620.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 617.25 | 618.05 | 620.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 614.00 | 614.72 | 617.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 586.39 | 594.24 | 601.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 596.10 | 594.23 | 600.33 | SL hit (close>ema200) qty=0.50 sl=594.23 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 591.95 | 586.96 | 586.70 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 578.50 | 585.71 | 586.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 574.85 | 579.84 | 582.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 578.05 | 575.21 | 577.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 578.05 | 575.21 | 577.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 578.05 | 575.21 | 577.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 578.05 | 575.21 | 577.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 579.90 | 576.15 | 577.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 581.05 | 576.15 | 577.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 574.75 | 575.87 | 577.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 573.90 | 575.87 | 577.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:00:00 | 573.50 | 572.76 | 574.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 574.45 | 573.21 | 574.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:00:00 | 573.90 | 573.61 | 574.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 574.50 | 573.78 | 574.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 574.60 | 573.78 | 574.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 573.40 | 573.71 | 574.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:30:00 | 573.05 | 573.62 | 574.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 573.00 | 573.72 | 574.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 573.15 | 573.60 | 574.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 571.80 | 573.38 | 573.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 574.30 | 573.57 | 573.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 574.30 | 573.57 | 573.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 574.75 | 573.80 | 573.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 574.70 | 573.80 | 573.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 574.20 | 573.88 | 573.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 574.75 | 573.88 | 573.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-13 15:15:00 | 574.80 | 574.07 | 574.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 574.80 | 574.07 | 574.05 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 572.65 | 573.78 | 573.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 571.00 | 573.23 | 573.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 574.95 | 571.41 | 572.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 574.95 | 571.41 | 572.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 574.95 | 571.41 | 572.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 574.20 | 571.41 | 572.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 566.85 | 570.67 | 571.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:45:00 | 566.60 | 569.94 | 571.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 566.65 | 569.24 | 570.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 566.50 | 568.81 | 570.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 578.80 | 571.57 | 571.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 578.80 | 571.57 | 571.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 580.00 | 573.82 | 572.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 580.15 | 580.21 | 577.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:45:00 | 580.05 | 580.21 | 577.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 578.85 | 580.06 | 578.46 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 565.55 | 576.38 | 577.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 564.50 | 569.76 | 573.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 563.55 | 562.64 | 565.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 11:30:00 | 563.70 | 562.64 | 565.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 566.00 | 563.58 | 565.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:45:00 | 565.70 | 563.58 | 565.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 560.50 | 562.96 | 565.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:15:00 | 560.20 | 562.96 | 565.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 562.00 | 557.65 | 557.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 562.00 | 557.65 | 557.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 563.35 | 558.79 | 557.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 559.30 | 560.89 | 559.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 559.30 | 560.89 | 559.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 559.30 | 560.89 | 559.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 559.30 | 560.89 | 559.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 557.60 | 560.24 | 559.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 557.60 | 560.24 | 559.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 556.00 | 558.82 | 558.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 554.20 | 556.84 | 557.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 555.85 | 554.44 | 555.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 555.85 | 554.44 | 555.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 555.85 | 554.44 | 555.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 555.85 | 554.44 | 555.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 553.75 | 554.30 | 555.60 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 560.85 | 556.86 | 556.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 10:15:00 | 561.80 | 559.46 | 558.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 558.25 | 559.22 | 558.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 11:15:00 | 558.25 | 559.22 | 558.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 558.25 | 559.22 | 558.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 558.25 | 559.22 | 558.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 558.50 | 559.08 | 558.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 558.30 | 559.08 | 558.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 558.50 | 558.96 | 558.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 557.90 | 558.96 | 558.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 557.70 | 558.71 | 558.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 557.70 | 558.71 | 558.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 558.80 | 558.73 | 558.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 559.60 | 558.73 | 558.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 580.35 | 585.97 | 586.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 580.35 | 585.97 | 586.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 579.15 | 581.74 | 583.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 571.80 | 568.86 | 572.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 571.80 | 568.86 | 572.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 568.60 | 568.16 | 570.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 564.30 | 567.19 | 569.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 573.90 | 568.81 | 568.96 | SL hit (close>static) qty=1.00 sl=571.20 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 575.95 | 570.24 | 569.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 576.85 | 571.56 | 570.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 575.70 | 576.98 | 574.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 575.70 | 576.98 | 574.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 575.50 | 576.52 | 574.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:30:00 | 577.55 | 576.64 | 575.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 577.00 | 576.65 | 575.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 573.75 | 576.13 | 575.28 | SL hit (close<static) qty=1.00 sl=574.40 alert=retest2 |

### Cycle 104 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 569.90 | 574.19 | 574.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 568.75 | 573.10 | 573.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 564.55 | 563.78 | 566.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 12:00:00 | 564.55 | 563.78 | 566.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 564.50 | 562.57 | 564.93 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 568.50 | 566.16 | 566.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 10:15:00 | 571.40 | 568.04 | 567.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 568.00 | 570.12 | 568.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 568.00 | 570.12 | 568.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 568.00 | 570.12 | 568.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 568.00 | 570.12 | 568.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 563.75 | 568.84 | 568.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 563.75 | 568.84 | 568.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 561.40 | 567.36 | 567.73 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 572.20 | 567.62 | 567.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 576.15 | 570.35 | 569.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 572.90 | 573.75 | 571.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 572.90 | 573.75 | 571.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 579.15 | 580.79 | 578.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 578.40 | 580.79 | 578.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 579.80 | 580.59 | 578.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 578.00 | 580.59 | 578.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 584.90 | 581.52 | 579.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 587.45 | 583.28 | 581.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 586.60 | 584.23 | 582.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 586.75 | 584.23 | 582.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 586.90 | 584.56 | 582.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 576.40 | 587.72 | 586.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 576.40 | 587.72 | 586.24 | SL hit (close<static) qty=1.00 sl=578.40 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 573.30 | 584.84 | 585.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 570.00 | 581.87 | 583.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 574.65 | 572.89 | 575.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 574.65 | 572.89 | 575.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 574.65 | 572.89 | 575.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 568.50 | 573.61 | 574.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 572.15 | 571.22 | 571.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 572.15 | 571.22 | 571.20 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 569.70 | 571.15 | 571.19 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 571.95 | 571.31 | 571.26 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 566.10 | 570.27 | 570.80 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 572.15 | 570.40 | 570.39 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 568.10 | 570.48 | 570.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 567.95 | 569.98 | 570.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 567.40 | 567.38 | 568.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 567.40 | 567.38 | 568.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 568.50 | 567.60 | 568.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 569.80 | 567.60 | 568.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 570.35 | 568.15 | 568.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 570.35 | 568.15 | 568.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 571.10 | 568.74 | 569.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:15:00 | 569.80 | 569.02 | 569.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:00:00 | 569.85 | 569.19 | 569.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 13:15:00 | 571.55 | 569.66 | 569.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 571.55 | 569.66 | 569.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 571.80 | 570.34 | 569.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 567.35 | 569.74 | 569.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 567.35 | 569.74 | 569.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 567.35 | 569.74 | 569.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 567.50 | 569.74 | 569.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 567.70 | 569.34 | 569.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 565.40 | 567.76 | 568.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 551.40 | 547.69 | 550.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 551.40 | 547.69 | 550.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 551.40 | 547.69 | 550.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 551.40 | 547.69 | 550.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 547.95 | 547.74 | 550.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 550.65 | 547.74 | 550.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 548.30 | 548.31 | 549.85 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 555.70 | 551.49 | 550.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 556.95 | 552.58 | 551.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 552.15 | 553.86 | 552.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 552.15 | 553.86 | 552.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 552.15 | 553.86 | 552.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:45:00 | 552.45 | 553.86 | 552.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 550.00 | 553.09 | 552.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 550.20 | 553.09 | 552.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 549.55 | 551.81 | 551.97 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 552.90 | 551.34 | 551.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 10:15:00 | 555.00 | 552.07 | 551.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 547.00 | 552.59 | 552.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 547.00 | 552.59 | 552.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 547.00 | 552.59 | 552.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 547.00 | 552.59 | 552.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 549.45 | 551.96 | 552.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 543.00 | 546.99 | 548.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 532.35 | 530.16 | 535.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 532.35 | 530.16 | 535.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 534.55 | 531.81 | 535.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 533.25 | 531.81 | 535.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 537.15 | 533.40 | 535.30 | SL hit (close>static) qty=1.00 sl=535.90 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 528.55 | 527.66 | 527.60 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 526.65 | 527.46 | 527.52 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 530.00 | 527.97 | 527.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 532.45 | 528.86 | 528.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 533.70 | 534.18 | 532.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 533.70 | 534.18 | 532.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 534.45 | 534.40 | 532.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 533.55 | 534.40 | 532.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 538.90 | 539.54 | 537.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 539.50 | 539.54 | 537.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 536.55 | 539.06 | 537.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 536.55 | 539.06 | 537.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 535.30 | 538.31 | 537.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 535.30 | 538.31 | 537.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 539.40 | 537.93 | 537.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 538.95 | 537.93 | 537.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 538.25 | 537.99 | 537.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 536.25 | 537.99 | 537.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 537.85 | 537.96 | 537.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 537.40 | 537.96 | 537.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 538.20 | 538.01 | 537.73 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 535.45 | 537.45 | 537.52 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 541.35 | 537.95 | 537.61 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 535.85 | 537.36 | 537.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 534.85 | 536.68 | 537.14 | Break + close below crossover candle low |

### Cycle 127 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 543.45 | 537.71 | 537.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 546.55 | 539.48 | 538.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 542.60 | 543.49 | 541.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 11:00:00 | 542.60 | 543.49 | 541.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 542.95 | 543.38 | 541.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 544.10 | 541.87 | 541.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 543.45 | 542.03 | 541.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 13:15:00 | 539.40 | 541.42 | 541.33 | SL hit (close<static) qty=1.00 sl=541.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 540.00 | 541.14 | 541.20 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 542.90 | 541.47 | 541.34 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 539.55 | 541.09 | 541.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 11:15:00 | 536.95 | 540.26 | 540.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 530.30 | 529.41 | 533.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:00:00 | 530.30 | 529.41 | 533.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 539.15 | 531.36 | 533.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 539.15 | 531.36 | 533.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 534.05 | 531.90 | 533.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:30:00 | 535.95 | 531.90 | 533.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 535.40 | 532.60 | 534.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:00:00 | 533.30 | 532.74 | 533.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 536.00 | 525.00 | 524.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 536.00 | 525.00 | 524.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 536.45 | 531.79 | 528.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 532.95 | 533.92 | 531.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 532.95 | 533.92 | 531.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 534.50 | 533.66 | 531.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 532.25 | 533.66 | 531.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 533.50 | 533.63 | 531.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 533.00 | 533.63 | 531.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 532.65 | 533.33 | 531.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:45:00 | 531.40 | 533.33 | 531.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 533.35 | 533.14 | 532.11 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 526.95 | 531.50 | 531.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 525.25 | 530.25 | 531.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 11:15:00 | 517.05 | 515.33 | 519.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 12:00:00 | 517.05 | 515.33 | 519.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 520.00 | 517.17 | 519.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 520.00 | 517.17 | 519.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 517.20 | 517.17 | 518.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 518.40 | 517.17 | 518.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 514.50 | 508.83 | 511.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 516.45 | 508.83 | 511.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 513.30 | 509.72 | 511.54 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 517.60 | 513.24 | 512.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 518.90 | 514.83 | 513.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 520.65 | 524.10 | 522.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 520.65 | 524.10 | 522.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 520.65 | 524.10 | 522.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 520.65 | 524.10 | 522.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 513.40 | 521.96 | 521.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 511.60 | 521.96 | 521.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 513.35 | 520.24 | 520.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 508.25 | 517.84 | 519.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 505.85 | 500.94 | 507.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 505.85 | 500.94 | 507.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 505.85 | 500.94 | 507.14 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 515.75 | 508.66 | 508.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 09:15:00 | 518.60 | 514.41 | 511.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 514.00 | 514.86 | 513.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 515.40 | 514.86 | 513.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 511.80 | 514.25 | 513.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 511.80 | 514.25 | 513.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 517.15 | 514.83 | 513.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 518.65 | 514.83 | 513.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 12:15:00 | 518.65 | 515.41 | 513.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:45:00 | 518.15 | 516.28 | 514.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 518.50 | 517.01 | 515.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 519.00 | 518.00 | 516.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:15:00 | 521.60 | 518.38 | 516.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 520.80 | 521.55 | 520.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:30:00 | 521.45 | 521.81 | 520.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:30:00 | 520.75 | 522.24 | 521.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 12:15:00 | 517.60 | 520.52 | 520.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 517.60 | 520.52 | 520.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 515.65 | 518.96 | 519.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 510.00 | 509.27 | 512.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:30:00 | 509.30 | 509.27 | 512.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 511.05 | 509.62 | 512.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 511.05 | 509.62 | 512.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 513.45 | 510.74 | 512.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 513.45 | 510.74 | 512.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 514.90 | 511.57 | 512.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 515.00 | 511.57 | 512.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 520.10 | 514.42 | 513.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 522.50 | 519.96 | 517.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 519.45 | 522.99 | 520.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 519.45 | 522.99 | 520.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 519.45 | 522.99 | 520.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 519.45 | 522.99 | 520.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 518.10 | 522.01 | 520.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 525.05 | 523.18 | 521.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 529.40 | 538.50 | 538.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 529.40 | 538.50 | 538.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 517.70 | 528.08 | 532.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 518.15 | 518.13 | 522.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:00:00 | 518.15 | 518.13 | 522.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 520.50 | 518.03 | 521.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 520.50 | 518.03 | 521.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 520.10 | 518.44 | 521.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 525.95 | 518.44 | 521.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 522.65 | 519.28 | 521.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 524.10 | 519.28 | 521.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 516.75 | 518.78 | 521.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 512.30 | 516.93 | 519.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 516.55 | 510.86 | 510.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 516.55 | 510.86 | 510.29 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 504.90 | 510.34 | 510.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 495.30 | 506.46 | 508.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 501.90 | 500.35 | 503.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 501.90 | 500.35 | 503.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 501.90 | 500.35 | 503.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 502.65 | 500.35 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 501.50 | 496.45 | 498.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 503.10 | 496.45 | 498.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 500.00 | 497.16 | 499.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 497.55 | 497.16 | 499.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 499.35 | 496.03 | 497.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 500.00 | 496.03 | 497.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 498.50 | 496.52 | 497.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 504.35 | 496.52 | 497.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 504.45 | 498.11 | 498.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 506.15 | 498.11 | 498.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 503.25 | 499.14 | 498.62 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 493.90 | 499.52 | 499.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 491.45 | 497.29 | 498.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 471.10 | 468.13 | 476.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 471.10 | 468.13 | 476.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 471.10 | 468.13 | 476.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 467.30 | 468.10 | 475.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 479.80 | 471.04 | 475.91 | SL hit (close>static) qty=1.00 sl=478.70 alert=retest2 |

### Cycle 143 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 495.00 | 478.57 | 478.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 497.90 | 482.44 | 479.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 11:15:00 | 502.55 | 503.75 | 497.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 11:45:00 | 502.30 | 503.75 | 497.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 498.00 | 502.21 | 498.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:00:00 | 498.00 | 502.21 | 498.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 495.90 | 500.95 | 497.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 495.90 | 500.95 | 497.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 496.30 | 500.02 | 497.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 510.50 | 500.02 | 497.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 523.20 | 526.62 | 526.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 523.20 | 526.62 | 526.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 519.80 | 525.25 | 526.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 530.00 | 524.44 | 525.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 530.00 | 524.44 | 525.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 530.00 | 524.44 | 525.23 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 528.00 | 525.93 | 525.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 15:15:00 | 530.05 | 527.08 | 526.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 537.65 | 538.57 | 536.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 535.75 | 537.75 | 536.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 535.75 | 537.75 | 536.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 552.30 | 537.75 | 536.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 545.40 | 551.34 | 552.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 545.40 | 551.34 | 552.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 545.00 | 550.07 | 551.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 13:15:00 | 544.15 | 539.40 | 542.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 13:15:00 | 544.15 | 539.40 | 542.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 544.15 | 539.40 | 542.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 544.15 | 539.40 | 542.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 543.05 | 540.13 | 542.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 541.15 | 540.80 | 542.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 547.05 | 542.05 | 542.83 | SL hit (close>static) qty=1.00 sl=545.40 alert=retest2 |

### Cycle 147 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 548.65 | 543.37 | 543.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 550.60 | 546.27 | 544.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 551.90 | 557.35 | 552.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 551.90 | 557.35 | 552.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 551.90 | 557.35 | 552.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 551.90 | 557.35 | 552.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 553.05 | 556.49 | 552.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 551.50 | 556.49 | 552.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 552.25 | 555.64 | 552.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:45:00 | 551.00 | 555.64 | 552.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 553.45 | 555.20 | 552.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 552.45 | 555.20 | 552.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 554.80 | 555.10 | 553.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 559.20 | 555.10 | 553.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 14:00:00 | 556.95 | 556.15 | 554.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 15:00:00 | 557.35 | 556.39 | 554.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:30:00 | 556.60 | 555.85 | 554.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 553.25 | 555.33 | 554.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 552.20 | 555.33 | 554.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 552.50 | 554.77 | 554.59 | SL hit (close<static) qty=1.00 sl=553.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 634.85 | 2024-05-24 15:15:00 | 648.80 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2024-05-16 11:30:00 | 634.45 | 2024-05-24 15:15:00 | 648.80 | STOP_HIT | 1.00 | 2.26% |
| BUY | retest2 | 2024-06-20 09:15:00 | 741.35 | 2024-06-21 12:15:00 | 729.40 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-06-20 12:45:00 | 739.05 | 2024-06-21 12:15:00 | 729.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-06-20 13:30:00 | 739.20 | 2024-06-21 12:15:00 | 729.40 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-06-20 14:00:00 | 739.05 | 2024-06-21 12:15:00 | 729.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-07-03 09:15:00 | 804.80 | 2024-07-04 09:15:00 | 792.65 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-07-03 11:15:00 | 797.60 | 2024-07-04 09:15:00 | 792.65 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-07-25 09:15:00 | 769.05 | 2024-07-26 12:15:00 | 793.45 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2024-08-01 11:15:00 | 768.40 | 2024-08-05 09:15:00 | 691.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 11:15:00 | 674.25 | 2024-08-26 09:15:00 | 664.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-08-23 12:45:00 | 674.00 | 2024-08-26 09:15:00 | 664.60 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-08-23 13:30:00 | 674.25 | 2024-08-26 09:15:00 | 664.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-09-05 09:15:00 | 706.45 | 2024-09-09 13:15:00 | 697.30 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-09-06 15:15:00 | 696.75 | 2024-09-09 13:15:00 | 697.30 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-09-09 09:45:00 | 696.45 | 2024-09-09 13:15:00 | 697.30 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2024-09-09 13:15:00 | 699.20 | 2024-09-09 13:15:00 | 697.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-09-19 10:15:00 | 665.50 | 2024-09-23 11:15:00 | 677.60 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-09-23 10:45:00 | 669.60 | 2024-09-23 11:15:00 | 677.60 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-09-27 12:45:00 | 665.00 | 2024-10-03 15:15:00 | 631.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 14:00:00 | 665.60 | 2024-10-03 15:15:00 | 632.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 13:15:00 | 665.30 | 2024-10-03 15:15:00 | 632.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 14:00:00 | 664.50 | 2024-10-03 15:15:00 | 631.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 12:45:00 | 665.00 | 2024-10-08 10:15:00 | 616.70 | STOP_HIT | 0.50 | 7.26% |
| SELL | retest2 | 2024-09-27 14:00:00 | 665.60 | 2024-10-08 10:15:00 | 616.70 | STOP_HIT | 0.50 | 7.35% |
| SELL | retest2 | 2024-09-30 13:15:00 | 665.30 | 2024-10-08 10:15:00 | 616.70 | STOP_HIT | 0.50 | 7.30% |
| SELL | retest2 | 2024-09-30 14:00:00 | 664.50 | 2024-10-08 10:15:00 | 616.70 | STOP_HIT | 0.50 | 7.19% |
| SELL | retest2 | 2024-10-09 11:45:00 | 619.95 | 2024-10-15 09:15:00 | 622.85 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-10-10 10:45:00 | 621.25 | 2024-10-15 09:15:00 | 622.85 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-10-23 15:15:00 | 598.85 | 2024-10-24 09:15:00 | 606.30 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-10-25 09:30:00 | 598.15 | 2024-10-28 10:15:00 | 613.10 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-10-25 13:15:00 | 598.15 | 2024-10-28 10:15:00 | 613.10 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-10-25 13:45:00 | 596.60 | 2024-10-28 10:15:00 | 613.10 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-12-16 11:15:00 | 624.05 | 2024-12-19 09:15:00 | 592.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 621.05 | 2024-12-19 09:15:00 | 590.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:15:00 | 624.05 | 2024-12-20 09:15:00 | 595.50 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2024-12-17 09:15:00 | 621.05 | 2024-12-20 09:15:00 | 595.50 | STOP_HIT | 0.50 | 4.11% |
| BUY | retest2 | 2025-01-02 13:45:00 | 606.85 | 2025-01-06 10:15:00 | 597.55 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-01-03 09:15:00 | 611.65 | 2025-01-06 10:15:00 | 597.55 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-01-06 09:30:00 | 606.15 | 2025-01-06 10:15:00 | 597.55 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-01-06 10:00:00 | 606.25 | 2025-01-06 10:15:00 | 597.55 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-01-29 09:15:00 | 579.05 | 2025-02-03 09:15:00 | 565.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-01-30 10:15:00 | 572.40 | 2025-02-03 09:15:00 | 565.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-01-30 11:45:00 | 574.40 | 2025-02-03 09:15:00 | 565.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-02-13 13:15:00 | 550.30 | 2025-02-17 09:15:00 | 522.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 550.40 | 2025-02-17 09:15:00 | 522.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 550.30 | 2025-02-17 13:15:00 | 535.40 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2025-02-13 15:00:00 | 550.40 | 2025-02-17 13:15:00 | 535.40 | STOP_HIT | 0.50 | 2.73% |
| BUY | retest2 | 2025-02-21 15:00:00 | 544.90 | 2025-02-24 10:15:00 | 537.40 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-03-11 10:45:00 | 527.60 | 2025-03-12 11:15:00 | 520.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-03-11 11:30:00 | 527.65 | 2025-03-12 11:15:00 | 520.95 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-03-11 12:30:00 | 528.90 | 2025-03-12 11:15:00 | 520.95 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-03-17 11:45:00 | 521.00 | 2025-03-18 09:15:00 | 533.25 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-03-17 14:15:00 | 520.75 | 2025-03-18 09:15:00 | 533.25 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-04-11 09:15:00 | 565.00 | 2025-04-24 09:15:00 | 621.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 606.95 | 2025-05-05 13:15:00 | 609.85 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-05-08 11:30:00 | 591.85 | 2025-05-12 12:15:00 | 595.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-05-12 10:00:00 | 591.15 | 2025-05-12 12:15:00 | 595.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-05-14 15:00:00 | 618.60 | 2025-05-19 11:15:00 | 604.85 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-15 12:45:00 | 617.00 | 2025-05-19 11:15:00 | 604.85 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-05-15 13:15:00 | 617.40 | 2025-05-19 11:15:00 | 604.85 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-05-16 10:30:00 | 618.60 | 2025-05-19 11:15:00 | 604.85 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-28 09:15:00 | 602.15 | 2025-05-28 10:15:00 | 598.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-05-30 10:30:00 | 588.60 | 2025-06-02 09:15:00 | 598.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-05-30 11:45:00 | 588.05 | 2025-06-02 09:15:00 | 598.95 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-06-05 10:00:00 | 606.85 | 2025-06-05 15:15:00 | 599.55 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-11 10:30:00 | 626.10 | 2025-06-12 11:15:00 | 612.35 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-06-11 11:15:00 | 625.25 | 2025-06-12 11:15:00 | 612.35 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-06-17 12:30:00 | 603.85 | 2025-06-24 09:15:00 | 604.75 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-06-18 11:00:00 | 603.50 | 2025-06-24 09:15:00 | 604.75 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-06-18 12:15:00 | 602.60 | 2025-06-24 09:15:00 | 604.75 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-06-27 14:45:00 | 612.50 | 2025-07-01 11:15:00 | 609.40 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-30 09:15:00 | 619.30 | 2025-07-01 11:15:00 | 609.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-07-01 09:45:00 | 612.75 | 2025-07-01 11:15:00 | 609.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-11 10:45:00 | 600.85 | 2025-07-14 09:15:00 | 610.25 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-07-24 10:30:00 | 617.25 | 2025-07-29 10:15:00 | 586.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 617.25 | 2025-07-29 12:15:00 | 596.10 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-07-25 09:30:00 | 614.00 | 2025-07-31 09:15:00 | 583.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 614.00 | 2025-08-04 09:15:00 | 585.20 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-08-08 10:15:00 | 573.90 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-11 11:00:00 | 573.50 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-08-11 13:00:00 | 574.45 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-08-11 15:00:00 | 573.90 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-12 10:30:00 | 573.05 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-08-12 14:00:00 | 573.00 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-08-12 15:00:00 | 573.15 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-08-13 11:45:00 | 571.80 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-08-18 12:45:00 | 566.60 | 2025-08-19 10:15:00 | 578.80 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-08-18 13:45:00 | 566.65 | 2025-08-19 10:15:00 | 578.80 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-08-18 14:30:00 | 566.50 | 2025-08-19 10:15:00 | 578.80 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-08-26 15:15:00 | 560.20 | 2025-09-01 13:15:00 | 562.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-09-10 09:15:00 | 559.60 | 2025-09-23 10:15:00 | 580.35 | STOP_HIT | 1.00 | 3.71% |
| SELL | retest2 | 2025-09-30 13:00:00 | 564.30 | 2025-10-01 12:15:00 | 573.90 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-10-06 13:30:00 | 577.55 | 2025-10-07 09:15:00 | 573.75 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-10-06 15:15:00 | 577.00 | 2025-10-07 09:15:00 | 573.75 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-28 09:15:00 | 587.45 | 2025-10-30 09:15:00 | 576.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-10-28 12:45:00 | 586.60 | 2025-10-30 09:15:00 | 576.40 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-10-28 13:15:00 | 586.75 | 2025-10-30 09:15:00 | 576.40 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-10-28 15:15:00 | 586.90 | 2025-10-30 09:15:00 | 576.40 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-11-06 09:15:00 | 568.50 | 2025-11-10 10:15:00 | 572.15 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-17 12:15:00 | 569.80 | 2025-11-17 13:15:00 | 571.55 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-11-17 13:00:00 | 569.85 | 2025-11-17 13:15:00 | 571.55 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-09 14:15:00 | 533.25 | 2025-12-10 09:15:00 | 537.15 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-10 10:45:00 | 533.05 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-12-11 10:45:00 | 533.40 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-12-11 11:30:00 | 532.50 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-12-12 10:15:00 | 533.25 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-12-12 14:15:00 | 533.00 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2026-01-06 09:45:00 | 544.10 | 2026-01-06 13:15:00 | 539.40 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-06 11:30:00 | 543.45 | 2026-01-06 13:15:00 | 539.40 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-09 14:00:00 | 533.30 | 2026-01-14 10:15:00 | 536.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-06 11:15:00 | 518.65 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-02-06 12:15:00 | 518.65 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-02-06 13:45:00 | 518.15 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-02-06 14:30:00 | 518.50 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-02-09 13:15:00 | 521.60 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-02-11 10:15:00 | 520.80 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-02-11 11:30:00 | 521.45 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-02-12 10:30:00 | 520.75 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-02-20 09:30:00 | 525.05 | 2026-03-02 09:15:00 | 529.40 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2026-03-06 14:45:00 | 512.30 | 2026-03-11 09:15:00 | 516.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-24 10:30:00 | 467.30 | 2026-03-24 12:15:00 | 479.80 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-04-01 09:15:00 | 510.50 | 2026-04-13 11:15:00 | 523.20 | STOP_HIT | 1.00 | 2.49% |
| BUY | retest2 | 2026-04-21 09:15:00 | 552.30 | 2026-04-23 14:15:00 | 545.40 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-04-28 09:15:00 | 541.15 | 2026-04-28 09:15:00 | 547.05 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-05-04 09:15:00 | 559.20 | 2026-05-05 11:15:00 | 552.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-05-04 14:00:00 | 556.95 | 2026-05-05 11:15:00 | 552.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-05-04 15:00:00 | 557.35 | 2026-05-05 11:15:00 | 552.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-05-05 09:30:00 | 556.60 | 2026-05-05 11:15:00 | 552.50 | STOP_HIT | 1.00 | -0.74% |
