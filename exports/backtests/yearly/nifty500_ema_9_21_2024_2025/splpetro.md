# Supreme Petrochem Ltd. (SPLPETRO)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 738.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 127 |
| ALERT1 | 88 |
| ALERT2 | 87 |
| ALERT2_SKIP | 46 |
| ALERT3 | 224 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 126 |
| PARTIAL | 13 |
| TARGET_HIT | 16 |
| STOP_HIT | 113 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 95
- **Target hits / Stop hits / Partials:** 16 / 112 / 13
- **Avg / median % per leg:** 0.90% / -0.91%
- **Sum % (uncompounded):** 126.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 15 | 26.8% | 10 | 46 | 0 | 0.54% | 30.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.21% | 0.4% |
| BUY @ 3rd Alert (retest2) | 54 | 13 | 24.1% | 10 | 44 | 0 | 0.56% | 30.1% |
| SELL (all) | 85 | 31 | 36.5% | 6 | 66 | 13 | 1.13% | 95.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 85 | 31 | 36.5% | 6 | 66 | 13 | 1.13% | 95.9% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.21% | 0.4% |
| retest2 (combined) | 139 | 44 | 31.7% | 16 | 110 | 13 | 0.91% | 126.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 691.70 | 703.81 | 705.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 10:15:00 | 684.10 | 699.87 | 703.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 13:15:00 | 698.70 | 696.60 | 700.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 13:15:00 | 698.70 | 696.60 | 700.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 698.70 | 696.60 | 700.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:45:00 | 699.20 | 696.60 | 700.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 689.40 | 693.39 | 698.11 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 707.05 | 698.42 | 697.55 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 688.15 | 695.83 | 696.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 663.20 | 673.83 | 678.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 15:15:00 | 652.75 | 652.50 | 657.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 15:15:00 | 652.75 | 652.50 | 657.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 652.75 | 652.50 | 657.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 657.25 | 652.50 | 657.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 656.80 | 653.36 | 657.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 655.75 | 653.25 | 656.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:00:00 | 655.75 | 653.75 | 656.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:45:00 | 655.05 | 654.27 | 656.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:00:00 | 655.25 | 654.47 | 656.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 658.65 | 655.30 | 656.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 15:00:00 | 658.65 | 655.30 | 656.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 665.05 | 657.25 | 657.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 638.00 | 657.25 | 657.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 09:15:00 | 669.30 | 654.22 | 653.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 669.30 | 654.22 | 653.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 684.65 | 660.31 | 656.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 697.55 | 698.16 | 688.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 697.55 | 698.16 | 688.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 714.35 | 704.29 | 695.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:15:00 | 735.65 | 704.29 | 695.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 10:15:00 | 724.55 | 732.11 | 732.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 724.55 | 732.11 | 732.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 724.10 | 728.14 | 730.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 740.35 | 729.92 | 730.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 11:15:00 | 740.35 | 729.92 | 730.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 740.35 | 729.92 | 730.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:00:00 | 740.35 | 729.92 | 730.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 12:15:00 | 759.50 | 735.84 | 733.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 775.00 | 753.74 | 743.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 15:15:00 | 795.00 | 795.66 | 781.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 09:15:00 | 788.70 | 795.66 | 781.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 798.60 | 796.25 | 783.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:15:00 | 812.95 | 796.25 | 783.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 776.00 | 785.29 | 786.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 776.00 | 785.29 | 786.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 15:15:00 | 775.50 | 781.85 | 784.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 784.55 | 781.86 | 783.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 784.55 | 781.86 | 783.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 784.55 | 781.86 | 783.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:45:00 | 786.65 | 781.86 | 783.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 786.00 | 782.69 | 784.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 786.00 | 782.69 | 784.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 777.95 | 781.74 | 783.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:45:00 | 773.70 | 779.86 | 781.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 14:45:00 | 774.00 | 776.95 | 780.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 767.80 | 777.70 | 780.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 776.70 | 770.01 | 772.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 780.90 | 772.19 | 773.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:30:00 | 784.10 | 772.19 | 773.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-01 14:15:00 | 777.70 | 774.12 | 774.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 14:15:00 | 777.70 | 774.12 | 774.01 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 772.00 | 773.70 | 773.83 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 780.20 | 775.00 | 774.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 10:15:00 | 787.90 | 780.89 | 778.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 785.25 | 785.36 | 782.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 785.25 | 785.36 | 782.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 785.25 | 785.36 | 782.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:00:00 | 797.55 | 788.36 | 785.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:15:00 | 796.95 | 788.89 | 785.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-09 09:15:00 | 877.31 | 859.73 | 838.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 861.40 | 869.53 | 869.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 13:15:00 | 858.50 | 864.62 | 867.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 874.85 | 863.68 | 865.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 874.85 | 863.68 | 865.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 874.85 | 863.68 | 865.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:15:00 | 875.55 | 863.68 | 865.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 892.00 | 869.35 | 868.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 11:15:00 | 920.00 | 879.48 | 872.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 895.15 | 896.20 | 886.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 895.15 | 896.20 | 886.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 873.00 | 894.88 | 890.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 873.00 | 894.88 | 890.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 863.65 | 888.63 | 887.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 863.65 | 888.63 | 887.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 874.15 | 885.74 | 886.72 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 14:15:00 | 890.00 | 887.69 | 887.47 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 10:15:00 | 880.00 | 886.30 | 886.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 11:15:00 | 877.45 | 881.85 | 883.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 12:15:00 | 890.05 | 883.49 | 884.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 890.05 | 883.49 | 884.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 890.05 | 883.49 | 884.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:00:00 | 890.05 | 883.49 | 884.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 892.30 | 885.25 | 885.12 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 10:15:00 | 883.00 | 885.70 | 886.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 875.50 | 881.17 | 883.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 887.85 | 881.66 | 883.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 887.85 | 881.66 | 883.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 887.85 | 881.66 | 883.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 887.85 | 881.66 | 883.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 888.70 | 883.07 | 883.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:30:00 | 893.00 | 883.07 | 883.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 882.85 | 883.42 | 883.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:30:00 | 880.55 | 881.28 | 882.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 09:30:00 | 881.55 | 879.18 | 881.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 10:30:00 | 881.70 | 878.72 | 880.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 10:00:00 | 876.05 | 872.82 | 876.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 878.95 | 874.04 | 876.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 882.00 | 874.04 | 876.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 880.55 | 875.34 | 876.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 13:00:00 | 877.40 | 875.76 | 876.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 15:00:00 | 876.20 | 876.35 | 877.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 866.90 | 876.88 | 877.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 836.52 | 851.71 | 858.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 837.47 | 851.71 | 858.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 837.62 | 851.71 | 858.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 832.25 | 851.71 | 858.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 833.53 | 851.71 | 858.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 832.39 | 851.71 | 858.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 823.55 | 851.71 | 858.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 829.00 | 824.46 | 837.98 | SL hit (close>ema200) qty=0.50 sl=824.46 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 827.85 | 820.88 | 819.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 15:15:00 | 835.00 | 823.30 | 821.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 11:15:00 | 821.95 | 825.07 | 822.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 11:15:00 | 821.95 | 825.07 | 822.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 821.95 | 825.07 | 822.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:00:00 | 821.95 | 825.07 | 822.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 824.65 | 824.99 | 823.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 13:30:00 | 827.75 | 825.75 | 823.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 15:15:00 | 828.80 | 839.89 | 840.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 15:15:00 | 828.80 | 839.89 | 840.33 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 846.40 | 840.77 | 840.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 849.25 | 843.56 | 841.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 878.55 | 880.63 | 869.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 878.55 | 880.63 | 869.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 883.75 | 888.76 | 884.40 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 880.00 | 882.77 | 882.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 874.75 | 881.16 | 882.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 888.70 | 870.70 | 873.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 888.70 | 870.70 | 873.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 888.70 | 870.70 | 873.78 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 11:15:00 | 894.50 | 878.76 | 877.11 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 13:15:00 | 875.45 | 879.71 | 879.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 859.55 | 872.49 | 876.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 867.85 | 866.81 | 871.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 867.85 | 866.81 | 871.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 861.60 | 865.63 | 870.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:15:00 | 860.00 | 864.79 | 869.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 858.75 | 863.52 | 867.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 09:15:00 | 878.90 | 866.01 | 867.90 | SL hit (close>static) qty=1.00 sl=875.50 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 883.60 | 870.52 | 868.78 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 11:15:00 | 871.60 | 874.51 | 874.60 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 11:15:00 | 882.00 | 875.21 | 874.42 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 859.60 | 872.68 | 873.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 850.35 | 866.19 | 870.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 862.00 | 860.90 | 866.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 862.90 | 860.90 | 866.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 864.60 | 861.64 | 865.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 10:15:00 | 858.55 | 861.64 | 865.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:00:00 | 858.25 | 860.96 | 865.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:45:00 | 854.55 | 859.47 | 864.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 14:45:00 | 855.00 | 858.72 | 860.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 861.00 | 859.17 | 860.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 859.05 | 859.17 | 860.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 855.10 | 858.36 | 859.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:30:00 | 861.70 | 858.36 | 859.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 857.60 | 858.01 | 859.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 858.05 | 858.01 | 859.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 857.90 | 857.44 | 858.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 857.75 | 857.44 | 858.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 855.70 | 857.09 | 858.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-16 11:15:00 | 867.00 | 858.61 | 858.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 11:15:00 | 867.00 | 858.61 | 858.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 14:15:00 | 872.40 | 862.71 | 860.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 872.20 | 872.81 | 868.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 09:45:00 | 875.05 | 872.81 | 868.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 879.85 | 874.06 | 870.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 870.10 | 874.06 | 870.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 872.30 | 875.33 | 872.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 872.30 | 875.33 | 872.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 870.80 | 874.42 | 871.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:15:00 | 871.45 | 874.42 | 871.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 876.05 | 874.75 | 872.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 871.85 | 874.75 | 872.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 872.35 | 874.27 | 872.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 872.35 | 874.27 | 872.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 871.20 | 873.66 | 872.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:30:00 | 873.40 | 873.66 | 872.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 880.00 | 874.92 | 872.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 890.00 | 874.92 | 872.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 12:30:00 | 882.00 | 877.83 | 875.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 861.85 | 879.65 | 877.54 | SL hit (close<static) qty=1.00 sl=870.10 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 12:15:00 | 868.70 | 875.05 | 875.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 13:15:00 | 866.00 | 873.24 | 874.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 14:15:00 | 874.05 | 873.40 | 874.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 14:15:00 | 874.05 | 873.40 | 874.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 874.05 | 873.40 | 874.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 15:15:00 | 865.00 | 873.40 | 874.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 12:00:00 | 865.40 | 868.35 | 870.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 14:15:00 | 879.80 | 871.85 | 871.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 879.80 | 871.85 | 871.49 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 866.60 | 870.91 | 871.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 865.65 | 869.17 | 870.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 876.00 | 869.52 | 869.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 876.00 | 869.52 | 869.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 876.00 | 869.52 | 869.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 875.10 | 869.52 | 869.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 875.30 | 870.68 | 870.39 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 865.80 | 870.05 | 870.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 862.00 | 867.75 | 868.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 787.65 | 776.88 | 793.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 787.65 | 776.88 | 793.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 787.65 | 776.88 | 793.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 789.00 | 776.88 | 793.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 784.50 | 783.41 | 789.40 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 15:15:00 | 796.85 | 792.63 | 792.23 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 788.05 | 791.87 | 792.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 14:15:00 | 786.70 | 790.84 | 791.69 | Break + close below crossover candle low |

### Cycle 36 — BUY (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 15:15:00 | 799.95 | 792.66 | 792.44 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 780.25 | 790.18 | 791.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 14:15:00 | 778.50 | 783.97 | 787.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 783.25 | 782.12 | 785.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-15 10:45:00 | 782.30 | 782.12 | 785.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 775.55 | 779.50 | 782.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 10:15:00 | 771.45 | 779.50 | 782.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 12:45:00 | 773.95 | 777.34 | 780.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 13:15:00 | 787.50 | 779.37 | 781.37 | SL hit (close>static) qty=1.00 sl=785.50 alert=retest2 |

### Cycle 38 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 796.95 | 782.89 | 782.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 15:15:00 | 809.40 | 788.19 | 785.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 784.65 | 787.48 | 785.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 784.65 | 787.48 | 785.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 784.65 | 787.48 | 785.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 784.65 | 787.48 | 785.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 776.70 | 785.32 | 784.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 776.70 | 785.32 | 784.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 775.05 | 783.27 | 783.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 767.05 | 776.79 | 780.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 776.30 | 775.37 | 778.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 776.30 | 775.37 | 778.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 775.20 | 775.33 | 777.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 775.20 | 775.33 | 777.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 780.00 | 776.27 | 778.09 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 791.45 | 780.86 | 779.95 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 773.65 | 781.82 | 782.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 11:15:00 | 766.45 | 778.74 | 780.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 779.75 | 778.54 | 780.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 14:00:00 | 779.75 | 778.54 | 780.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 781.85 | 779.36 | 780.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 777.90 | 779.36 | 780.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 769.95 | 777.48 | 779.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 761.25 | 774.23 | 777.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 15:15:00 | 757.00 | 748.99 | 748.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 757.00 | 748.99 | 748.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 759.00 | 750.99 | 749.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 14:15:00 | 761.20 | 762.38 | 756.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 14:15:00 | 761.20 | 762.38 | 756.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 761.20 | 762.38 | 756.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 761.20 | 762.38 | 756.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 765.60 | 763.12 | 758.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 10:15:00 | 774.35 | 766.16 | 761.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 11:00:00 | 775.80 | 768.09 | 763.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 10:45:00 | 773.80 | 773.95 | 769.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 11:15:00 | 773.65 | 773.95 | 769.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 773.25 | 773.81 | 769.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:30:00 | 773.45 | 773.81 | 769.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 768.60 | 772.77 | 769.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:30:00 | 770.20 | 772.77 | 769.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 767.85 | 771.79 | 769.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 767.80 | 771.79 | 769.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 761.05 | 769.64 | 768.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 761.05 | 769.64 | 768.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 753.75 | 765.56 | 766.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 753.75 | 765.56 | 766.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 736.55 | 752.35 | 758.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 706.75 | 705.20 | 717.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 706.75 | 705.20 | 717.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 699.75 | 692.31 | 698.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 697.50 | 692.31 | 698.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 700.00 | 693.85 | 698.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:30:00 | 699.05 | 693.90 | 697.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 698.45 | 690.99 | 690.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 698.45 | 690.99 | 690.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 705.15 | 693.82 | 691.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 12:15:00 | 706.10 | 707.91 | 702.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 13:00:00 | 706.10 | 707.91 | 702.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 700.05 | 706.34 | 702.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:00:00 | 700.05 | 706.34 | 702.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 706.10 | 706.29 | 702.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 709.00 | 705.27 | 702.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-09 09:15:00 | 779.90 | 768.35 | 763.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 13:15:00 | 770.95 | 774.04 | 774.27 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 15:15:00 | 775.95 | 774.55 | 774.47 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 774.25 | 774.43 | 774.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 771.60 | 773.86 | 774.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 730.35 | 719.55 | 730.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 730.35 | 719.55 | 730.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 730.35 | 719.55 | 730.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 694.95 | 708.79 | 713.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:45:00 | 698.90 | 703.04 | 709.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 663.95 | 670.34 | 678.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 660.20 | 667.06 | 676.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 661.45 | 660.71 | 668.43 | SL hit (close>ema200) qty=0.50 sl=660.71 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 649.10 | 644.19 | 643.57 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 642.25 | 646.35 | 646.36 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 652.00 | 647.48 | 646.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 15:15:00 | 661.40 | 650.62 | 648.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 642.55 | 649.01 | 647.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 642.55 | 649.01 | 647.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 642.55 | 649.01 | 647.89 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 642.00 | 646.99 | 647.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 11:15:00 | 639.90 | 645.57 | 646.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 10:15:00 | 642.90 | 641.79 | 643.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 11:00:00 | 642.90 | 641.79 | 643.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 584.65 | 577.46 | 585.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 596.50 | 583.59 | 587.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 597.65 | 586.40 | 588.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 595.05 | 586.40 | 588.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:45:00 | 595.05 | 588.36 | 588.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:15:00 | 594.55 | 588.36 | 588.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 12:15:00 | 597.00 | 590.09 | 589.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 597.00 | 590.09 | 589.56 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 13:15:00 | 583.00 | 588.67 | 588.96 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 603.45 | 591.37 | 590.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 607.15 | 596.97 | 593.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 15:15:00 | 599.10 | 599.19 | 595.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 12:00:00 | 611.65 | 603.14 | 598.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 13:30:00 | 610.90 | 605.07 | 599.95 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 622.55 | 618.94 | 612.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:30:00 | 626.45 | 619.53 | 613.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 11:15:00 | 612.55 | 618.14 | 613.05 | SL hit (close<ema400) qty=1.00 sl=613.05 alert=retest1 |

### Cycle 55 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 606.40 | 610.74 | 610.77 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 627.00 | 613.79 | 612.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 628.90 | 616.81 | 613.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 13:15:00 | 673.50 | 673.68 | 663.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 652.65 | 669.71 | 664.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 652.65 | 669.71 | 664.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 652.65 | 669.71 | 664.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 665.00 | 668.77 | 664.55 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 641.60 | 660.09 | 661.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 638.15 | 655.70 | 659.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 573.20 | 569.72 | 581.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-19 09:30:00 | 574.60 | 569.72 | 581.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 575.95 | 572.30 | 580.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:30:00 | 580.00 | 572.30 | 580.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 573.85 | 572.21 | 577.33 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 10:15:00 | 586.50 | 577.86 | 577.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 12:15:00 | 590.10 | 581.79 | 579.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 581.15 | 584.58 | 581.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 581.15 | 584.58 | 581.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 581.15 | 584.58 | 581.90 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 571.00 | 580.33 | 580.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 566.90 | 575.54 | 578.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 540.20 | 535.31 | 544.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 14:15:00 | 540.20 | 535.31 | 544.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 540.20 | 535.31 | 544.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 540.20 | 535.31 | 544.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 547.35 | 538.63 | 544.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:15:00 | 546.55 | 538.63 | 544.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 546.85 | 540.27 | 544.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:00:00 | 545.65 | 541.35 | 544.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:45:00 | 545.10 | 542.06 | 544.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 14:15:00 | 551.60 | 544.80 | 545.63 | SL hit (close>static) qty=1.00 sl=551.55 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 15:15:00 | 555.10 | 546.86 | 546.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 558.35 | 550.46 | 548.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 593.40 | 593.71 | 582.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:30:00 | 593.70 | 593.71 | 582.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 583.45 | 593.05 | 584.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 583.45 | 593.05 | 584.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 580.15 | 590.47 | 584.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 580.15 | 590.47 | 584.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 578.15 | 588.01 | 583.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 578.15 | 588.01 | 583.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 570.90 | 580.80 | 581.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 562.15 | 577.07 | 579.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 575.40 | 571.76 | 575.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 575.40 | 571.76 | 575.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 575.40 | 571.76 | 575.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:15:00 | 577.90 | 571.76 | 575.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 579.45 | 573.30 | 575.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 579.45 | 573.30 | 575.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 575.00 | 573.64 | 575.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 575.75 | 573.64 | 575.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 577.00 | 574.31 | 575.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:00:00 | 577.00 | 574.31 | 575.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 575.60 | 574.57 | 575.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:30:00 | 576.10 | 574.57 | 575.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 579.10 | 575.47 | 575.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:30:00 | 577.40 | 575.47 | 575.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 579.95 | 576.37 | 576.28 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 09:15:00 | 573.40 | 575.78 | 576.02 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 582.45 | 575.01 | 575.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 12:15:00 | 583.45 | 577.81 | 576.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 13:15:00 | 573.55 | 576.96 | 576.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 13:15:00 | 573.55 | 576.96 | 576.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 573.55 | 576.96 | 576.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 573.55 | 576.96 | 576.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 586.00 | 578.77 | 577.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 15:15:00 | 603.00 | 592.15 | 585.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 12:15:00 | 603.75 | 597.90 | 590.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-25 09:15:00 | 663.30 | 637.80 | 628.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 13:15:00 | 631.70 | 640.87 | 642.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 14:15:00 | 618.00 | 636.30 | 639.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 15:15:00 | 635.00 | 628.97 | 633.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 15:15:00 | 635.00 | 628.97 | 633.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 635.00 | 628.97 | 633.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 624.25 | 628.97 | 633.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 619.70 | 628.37 | 632.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 624.50 | 622.13 | 625.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:30:00 | 620.35 | 620.75 | 623.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 625.75 | 621.61 | 623.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:00:00 | 625.75 | 621.61 | 623.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 625.00 | 622.28 | 623.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 615.80 | 623.26 | 623.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 15:15:00 | 593.04 | 606.22 | 613.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 15:15:00 | 593.27 | 606.22 | 613.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 561.83 | 597.71 | 608.80 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 597.05 | 591.97 | 591.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 613.45 | 597.03 | 593.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 635.25 | 635.47 | 628.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 635.25 | 635.47 | 628.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 632.00 | 634.89 | 629.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:45:00 | 630.35 | 634.89 | 629.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 627.75 | 634.43 | 631.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:45:00 | 628.75 | 634.43 | 631.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 627.00 | 632.95 | 630.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 15:00:00 | 627.00 | 632.95 | 630.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 09:15:00 | 615.00 | 628.89 | 629.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 11:15:00 | 612.00 | 623.33 | 626.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 11:15:00 | 618.55 | 611.02 | 617.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 11:15:00 | 618.55 | 611.02 | 617.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 618.55 | 611.02 | 617.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 618.55 | 611.02 | 617.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 612.10 | 611.24 | 616.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-23 13:15:00 | 609.00 | 611.24 | 616.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 13:15:00 | 624.30 | 613.85 | 617.50 | SL hit (close>static) qty=1.00 sl=622.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 630.85 | 620.79 | 620.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 11:15:00 | 655.15 | 631.91 | 625.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 633.35 | 640.20 | 633.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 633.35 | 640.20 | 633.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 633.35 | 640.20 | 633.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:30:00 | 628.00 | 640.20 | 633.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 635.00 | 639.16 | 633.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 629.75 | 639.16 | 633.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 646.75 | 646.56 | 640.25 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 636.65 | 640.31 | 640.72 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 14:15:00 | 652.00 | 641.09 | 640.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 10:15:00 | 653.30 | 650.01 | 647.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 650.00 | 651.43 | 649.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 650.00 | 651.43 | 649.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 650.00 | 651.43 | 649.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 09:45:00 | 658.30 | 655.12 | 652.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 09:45:00 | 660.05 | 656.96 | 654.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-12 09:15:00 | 724.13 | 686.70 | 672.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 692.55 | 696.39 | 696.58 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 701.50 | 696.98 | 696.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 715.55 | 702.69 | 699.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 706.60 | 713.30 | 710.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 706.60 | 713.30 | 710.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 706.60 | 713.30 | 710.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 706.60 | 713.30 | 710.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 709.25 | 712.49 | 709.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:00:00 | 713.45 | 710.32 | 709.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 710.85 | 709.91 | 709.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:45:00 | 711.00 | 710.11 | 709.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 713.60 | 710.81 | 709.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 12:15:00 | 701.05 | 708.85 | 709.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 12:15:00 | 701.05 | 708.85 | 709.13 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 719.05 | 709.48 | 709.15 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 711.00 | 712.11 | 712.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 12:15:00 | 705.20 | 709.60 | 710.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 12:15:00 | 715.75 | 708.99 | 709.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 12:15:00 | 715.75 | 708.99 | 709.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 715.75 | 708.99 | 709.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 715.10 | 708.99 | 709.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 713.75 | 709.94 | 709.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:45:00 | 714.60 | 709.94 | 709.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 715.60 | 711.07 | 710.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 721.40 | 713.77 | 711.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 724.35 | 724.60 | 719.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:45:00 | 723.10 | 724.60 | 719.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 715.75 | 723.39 | 720.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:00:00 | 715.75 | 723.39 | 720.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 715.20 | 721.76 | 720.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:15:00 | 712.65 | 721.76 | 720.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 721.40 | 720.24 | 719.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:15:00 | 724.00 | 720.24 | 719.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 718.30 | 719.97 | 719.64 | SL hit (close<static) qty=1.00 sl=718.85 alert=retest2 |

### Cycle 77 — SELL (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 13:15:00 | 717.05 | 719.39 | 719.41 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 724.95 | 719.42 | 719.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 730.55 | 723.19 | 721.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 11:15:00 | 728.20 | 729.75 | 726.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 12:00:00 | 728.20 | 729.75 | 726.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 726.45 | 728.90 | 726.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 726.45 | 728.90 | 726.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 727.10 | 728.54 | 726.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:15:00 | 733.50 | 728.54 | 726.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 733.50 | 729.53 | 727.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 740.65 | 732.17 | 728.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 768.50 | 756.72 | 750.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 775.00 | 760.47 | 757.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 776.90 | 784.93 | 785.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 776.90 | 784.93 | 785.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 775.75 | 783.09 | 785.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 796.25 | 784.57 | 785.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 796.25 | 784.57 | 785.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 796.25 | 784.57 | 785.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 796.25 | 784.57 | 785.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 795.80 | 786.82 | 786.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 802.75 | 791.67 | 788.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 871.45 | 872.16 | 851.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:00:00 | 871.45 | 872.16 | 851.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 870.85 | 877.45 | 874.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 866.15 | 877.45 | 874.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 864.70 | 874.90 | 873.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 864.70 | 874.90 | 873.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 863.50 | 872.62 | 872.56 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 863.65 | 870.83 | 871.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 859.55 | 866.51 | 869.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 15:15:00 | 843.75 | 841.94 | 850.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 847.30 | 843.01 | 850.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 847.30 | 843.01 | 850.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 847.30 | 843.01 | 850.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 812.85 | 807.68 | 811.29 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 825.00 | 814.83 | 813.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 827.55 | 817.38 | 815.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 813.90 | 817.90 | 815.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 813.90 | 817.90 | 815.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 813.90 | 817.90 | 815.74 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 810.40 | 814.84 | 815.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 809.05 | 813.68 | 814.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 808.25 | 808.12 | 811.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 808.25 | 808.12 | 811.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 810.55 | 808.92 | 810.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 809.45 | 808.92 | 810.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 814.20 | 809.98 | 811.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:30:00 | 815.80 | 809.98 | 811.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 816.75 | 811.33 | 811.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 816.90 | 811.33 | 811.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 817.00 | 812.47 | 812.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 821.80 | 814.97 | 813.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 815.35 | 816.29 | 814.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 815.35 | 816.29 | 814.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 815.35 | 816.29 | 814.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 815.35 | 816.29 | 814.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 816.30 | 816.29 | 814.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 816.30 | 816.29 | 814.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 816.25 | 816.28 | 814.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 814.00 | 816.28 | 814.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 816.60 | 816.30 | 815.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 815.00 | 816.30 | 815.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 820.00 | 817.04 | 815.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:00:00 | 820.05 | 817.64 | 815.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 821.95 | 818.81 | 816.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 821.25 | 820.67 | 819.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:15:00 | 821.50 | 820.67 | 819.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 819.65 | 820.40 | 819.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 826.15 | 819.98 | 819.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 815.35 | 818.95 | 818.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 815.35 | 818.95 | 818.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 806.70 | 814.06 | 816.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 14:15:00 | 810.05 | 808.63 | 811.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 810.05 | 808.63 | 811.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 817.75 | 810.67 | 811.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 824.95 | 810.67 | 811.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 819.40 | 812.42 | 812.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 819.10 | 812.42 | 812.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 816.60 | 813.25 | 813.00 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 809.55 | 813.47 | 813.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 806.55 | 810.31 | 811.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 805.00 | 803.87 | 807.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 805.00 | 803.87 | 807.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 797.55 | 802.10 | 804.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 792.15 | 798.99 | 802.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:30:00 | 790.00 | 792.92 | 796.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 788.00 | 794.12 | 796.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 808.05 | 795.13 | 796.20 | SL hit (close>static) qty=1.00 sl=806.95 alert=retest2 |

### Cycle 88 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 817.00 | 799.51 | 798.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 11:15:00 | 821.05 | 803.82 | 800.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 12:15:00 | 817.35 | 823.67 | 814.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 817.35 | 823.67 | 814.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 817.35 | 823.67 | 814.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:45:00 | 814.55 | 823.67 | 814.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 807.60 | 820.45 | 814.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 809.15 | 820.45 | 814.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 815.00 | 819.36 | 814.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:45:00 | 822.55 | 819.33 | 815.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:30:00 | 822.10 | 819.27 | 815.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 805.05 | 815.11 | 815.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 805.05 | 815.11 | 815.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 12:15:00 | 802.75 | 811.25 | 813.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 752.75 | 751.55 | 764.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 12:00:00 | 752.75 | 751.55 | 764.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 741.00 | 737.18 | 742.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 741.00 | 737.18 | 742.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 752.05 | 740.16 | 743.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 756.00 | 740.16 | 743.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 758.00 | 743.73 | 744.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 747.30 | 743.73 | 744.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 751.00 | 738.58 | 740.63 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 775.00 | 748.10 | 744.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 800.80 | 758.64 | 749.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 828.30 | 829.09 | 812.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:30:00 | 828.25 | 829.09 | 812.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 822.30 | 827.12 | 820.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 819.15 | 827.12 | 820.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 814.55 | 824.61 | 820.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 814.65 | 824.61 | 820.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 811.85 | 822.06 | 819.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 811.85 | 822.06 | 819.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 797.00 | 815.42 | 816.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 788.00 | 807.21 | 812.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 765.25 | 759.89 | 769.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 765.25 | 759.89 | 769.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 762.65 | 762.77 | 768.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:45:00 | 762.90 | 762.77 | 768.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 765.00 | 763.22 | 767.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 769.30 | 763.22 | 767.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 771.90 | 764.95 | 768.31 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 776.50 | 770.09 | 769.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 778.10 | 771.69 | 770.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 776.10 | 779.96 | 776.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 776.10 | 779.96 | 776.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 776.10 | 779.96 | 776.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 774.05 | 779.96 | 776.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 772.70 | 778.51 | 776.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 773.35 | 778.51 | 776.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 778.10 | 778.42 | 776.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 779.05 | 778.42 | 776.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 773.40 | 775.24 | 775.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 773.40 | 775.24 | 775.33 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 780.35 | 776.26 | 775.79 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 775.95 | 780.38 | 780.86 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 784.15 | 780.91 | 780.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 786.05 | 782.61 | 781.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 780.65 | 782.51 | 781.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 780.65 | 782.51 | 781.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 780.65 | 782.51 | 781.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 780.65 | 782.51 | 781.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 780.40 | 782.09 | 781.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 14:45:00 | 791.20 | 783.82 | 782.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-26 09:15:00 | 870.32 | 860.98 | 857.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 840.55 | 872.68 | 876.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 831.05 | 864.35 | 872.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 09:15:00 | 835.65 | 827.77 | 839.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 835.65 | 827.77 | 839.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 835.65 | 827.77 | 839.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:45:00 | 840.40 | 827.77 | 839.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 841.50 | 830.45 | 838.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:45:00 | 842.65 | 830.45 | 838.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 845.15 | 833.39 | 838.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:30:00 | 843.85 | 833.39 | 838.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 846.35 | 839.42 | 840.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 842.65 | 839.42 | 840.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 840.40 | 839.62 | 840.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 837.90 | 839.62 | 840.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 09:15:00 | 796.00 | 810.70 | 820.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 793.35 | 793.24 | 802.06 | SL hit (close>ema200) qty=0.50 sl=793.24 alert=retest2 |

### Cycle 98 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 805.05 | 780.07 | 777.26 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 790.20 | 792.80 | 792.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 769.50 | 787.51 | 790.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 755.75 | 754.09 | 761.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 755.75 | 754.09 | 761.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 663.15 | 659.68 | 664.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 663.40 | 659.68 | 664.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 666.60 | 661.76 | 664.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 666.95 | 661.76 | 664.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 671.00 | 663.60 | 665.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 671.15 | 663.60 | 665.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 651.35 | 654.35 | 657.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:30:00 | 651.00 | 653.63 | 657.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 644.00 | 651.71 | 655.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 649.30 | 650.68 | 654.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:00:00 | 650.25 | 651.56 | 654.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 653.40 | 652.32 | 654.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:00:00 | 653.40 | 652.32 | 654.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 655.90 | 653.03 | 654.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 655.90 | 653.03 | 654.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 652.10 | 652.85 | 654.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 649.25 | 652.85 | 654.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 12:15:00 | 651.00 | 636.42 | 635.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 12:15:00 | 651.00 | 636.42 | 635.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 654.05 | 639.95 | 637.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 10:15:00 | 663.05 | 664.70 | 659.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 11:15:00 | 662.40 | 664.24 | 660.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 662.40 | 664.24 | 660.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 661.25 | 664.24 | 660.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 663.70 | 664.60 | 661.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:45:00 | 663.90 | 664.60 | 661.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 657.80 | 663.24 | 660.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 660.40 | 663.24 | 660.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 661.00 | 662.79 | 660.96 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 646.80 | 658.11 | 659.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 645.00 | 652.40 | 656.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 11:15:00 | 631.20 | 629.99 | 637.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 11:45:00 | 631.00 | 629.99 | 637.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 638.95 | 631.78 | 637.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 638.95 | 631.78 | 637.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 637.05 | 632.83 | 637.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 15:15:00 | 636.00 | 634.20 | 637.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 635.90 | 634.96 | 637.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 636.25 | 634.82 | 636.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 640.70 | 634.96 | 636.11 | SL hit (close>static) qty=1.00 sl=638.95 alert=retest2 |

### Cycle 102 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 646.10 | 632.76 | 630.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 679.45 | 642.10 | 635.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 658.00 | 663.49 | 652.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 15:00:00 | 658.00 | 663.49 | 652.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 658.30 | 661.12 | 653.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 660.05 | 661.12 | 653.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 660.00 | 660.71 | 653.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:45:00 | 663.90 | 661.17 | 654.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 660.20 | 662.20 | 657.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 658.15 | 661.39 | 657.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 658.15 | 661.39 | 657.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 656.40 | 660.39 | 657.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 657.05 | 660.39 | 657.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 656.90 | 659.69 | 657.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 657.65 | 659.69 | 657.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 657.00 | 659.15 | 657.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 656.25 | 659.15 | 657.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 657.30 | 658.78 | 657.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 656.00 | 658.78 | 657.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 655.40 | 658.11 | 657.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 655.40 | 658.11 | 657.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 650.00 | 656.49 | 656.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 15:15:00 | 650.00 | 656.49 | 656.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 649.85 | 652.32 | 654.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 13:15:00 | 641.65 | 640.61 | 644.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 14:00:00 | 641.65 | 640.61 | 644.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 641.05 | 640.68 | 643.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 637.60 | 640.68 | 643.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 648.30 | 634.85 | 635.58 | SL hit (close>static) qty=1.00 sl=644.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 646.20 | 637.12 | 636.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 649.15 | 643.43 | 640.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 643.25 | 645.65 | 642.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:00:00 | 643.25 | 645.65 | 642.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 644.30 | 645.38 | 642.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 644.30 | 645.38 | 642.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 640.50 | 644.41 | 642.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 656.20 | 644.41 | 642.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 639.65 | 644.81 | 645.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 639.65 | 644.81 | 645.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 635.55 | 640.09 | 641.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 640.35 | 639.13 | 640.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 640.35 | 639.13 | 640.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 640.35 | 639.13 | 640.56 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 644.55 | 641.51 | 641.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 649.80 | 643.76 | 642.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 645.10 | 645.21 | 643.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:15:00 | 644.60 | 645.21 | 643.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 644.40 | 645.05 | 643.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 645.65 | 645.05 | 643.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 644.25 | 644.89 | 643.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:30:00 | 648.00 | 645.13 | 644.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 648.45 | 646.07 | 644.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 647.70 | 646.42 | 645.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 642.55 | 645.07 | 644.92 | SL hit (close<static) qty=1.00 sl=643.60 alert=retest2 |

### Cycle 107 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 642.95 | 644.64 | 644.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 14:15:00 | 642.50 | 643.66 | 644.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 15:15:00 | 646.00 | 644.13 | 644.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 15:15:00 | 646.00 | 644.13 | 644.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 646.00 | 644.13 | 644.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 635.20 | 644.13 | 644.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 14:15:00 | 603.44 | 629.21 | 636.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 15:15:00 | 571.68 | 580.99 | 591.74 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 108 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 523.75 | 511.98 | 511.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 527.60 | 519.60 | 515.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 517.20 | 521.74 | 517.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 517.20 | 521.74 | 517.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 517.20 | 521.74 | 517.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 517.55 | 521.74 | 517.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 512.20 | 519.83 | 517.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 512.25 | 519.83 | 517.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 521.60 | 520.19 | 517.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:00:00 | 523.55 | 520.86 | 518.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 547.00 | 530.05 | 523.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-30 15:15:00 | 575.90 | 554.85 | 540.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 590.60 | 609.09 | 610.34 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 630.90 | 608.16 | 607.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 635.00 | 613.52 | 609.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 642.30 | 644.40 | 638.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 642.30 | 644.40 | 638.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 642.30 | 644.40 | 638.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 632.70 | 644.40 | 638.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 627.50 | 640.34 | 639.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 628.20 | 640.34 | 639.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 634.10 | 639.10 | 638.66 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 629.55 | 637.19 | 637.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 620.75 | 629.89 | 633.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 625.10 | 623.71 | 628.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 625.10 | 623.71 | 628.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 633.05 | 625.58 | 628.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:45:00 | 635.85 | 625.58 | 628.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 632.20 | 626.90 | 628.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:15:00 | 630.70 | 626.90 | 628.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:00:00 | 630.45 | 628.57 | 629.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 630.40 | 629.36 | 629.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 632.90 | 630.06 | 629.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 632.90 | 630.06 | 629.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 638.85 | 631.82 | 630.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 633.10 | 636.72 | 634.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 633.10 | 636.72 | 634.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 633.10 | 636.72 | 634.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 633.10 | 636.72 | 634.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 658.50 | 641.08 | 636.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:30:00 | 632.65 | 641.08 | 636.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 646.95 | 648.29 | 642.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 672.00 | 654.32 | 646.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 658.00 | 657.41 | 651.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 654.60 | 656.02 | 652.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 661.00 | 652.47 | 651.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 661.10 | 654.19 | 652.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 646.80 | 651.35 | 651.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 646.80 | 651.35 | 651.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 641.00 | 649.28 | 650.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 648.45 | 648.27 | 650.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 648.45 | 648.27 | 650.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 652.55 | 649.16 | 650.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 644.40 | 649.38 | 650.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 13:15:00 | 651.60 | 650.60 | 650.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 651.60 | 650.60 | 650.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 656.60 | 651.80 | 651.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 694.55 | 695.36 | 679.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 12:00:00 | 694.55 | 695.36 | 679.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 682.80 | 692.84 | 679.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:00:00 | 682.80 | 692.84 | 679.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 684.65 | 691.21 | 680.11 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 651.60 | 671.79 | 674.09 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 678.85 | 671.88 | 671.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 692.25 | 681.40 | 676.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 669.95 | 681.07 | 677.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 669.95 | 681.07 | 677.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 669.95 | 681.07 | 677.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:15:00 | 670.00 | 681.07 | 677.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 670.00 | 678.86 | 676.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 656.50 | 678.86 | 676.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 661.00 | 672.85 | 674.42 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 693.90 | 676.61 | 674.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 695.65 | 684.28 | 679.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 693.50 | 697.43 | 691.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 693.50 | 697.43 | 691.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 693.50 | 697.43 | 691.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 693.50 | 697.43 | 691.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 698.00 | 697.54 | 692.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 681.50 | 697.54 | 692.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 693.40 | 696.72 | 692.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 702.10 | 696.72 | 692.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 695.30 | 696.13 | 692.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 687.40 | 691.09 | 691.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 687.40 | 691.09 | 691.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 670.10 | 683.45 | 687.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 664.45 | 664.13 | 672.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 667.95 | 664.13 | 672.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 664.40 | 663.48 | 668.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 667.05 | 663.48 | 668.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 654.55 | 662.32 | 667.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:30:00 | 658.70 | 662.32 | 667.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 667.00 | 662.72 | 666.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 667.50 | 662.72 | 666.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 669.55 | 664.09 | 667.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 669.55 | 664.09 | 667.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 666.55 | 664.58 | 666.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 671.85 | 664.58 | 666.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 668.45 | 665.36 | 667.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 668.45 | 665.36 | 667.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 668.45 | 665.97 | 667.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:30:00 | 670.45 | 665.97 | 667.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 673.00 | 667.49 | 667.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 662.30 | 667.49 | 667.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 10:15:00 | 664.15 | 666.95 | 667.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 664.00 | 663.09 | 663.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:00:00 | 663.85 | 663.25 | 663.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 663.65 | 663.33 | 663.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 664.70 | 663.33 | 663.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 656.05 | 661.87 | 663.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 653.90 | 661.87 | 663.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:00:00 | 652.50 | 645.15 | 651.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 676.00 | 652.73 | 653.89 | SL hit (close>static) qty=1.00 sl=674.00 alert=retest2 |

### Cycle 120 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 691.60 | 660.51 | 657.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 714.90 | 682.29 | 674.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 13:15:00 | 740.45 | 741.24 | 726.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-01 13:45:00 | 740.45 | 741.24 | 726.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 721.00 | 736.19 | 727.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 731.25 | 730.84 | 727.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 15:00:00 | 734.95 | 731.66 | 727.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-06 09:15:00 | 804.38 | 740.10 | 732.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 726.55 | 736.08 | 737.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 14:15:00 | 723.90 | 732.66 | 735.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 738.80 | 731.15 | 734.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 738.80 | 731.15 | 734.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 738.80 | 731.15 | 734.04 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 745.90 | 736.54 | 736.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 758.65 | 742.65 | 739.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 12:15:00 | 746.50 | 747.30 | 743.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 13:00:00 | 746.50 | 747.30 | 743.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 746.00 | 746.39 | 743.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:15:00 | 743.00 | 746.39 | 743.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 743.00 | 745.71 | 743.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 744.65 | 745.71 | 743.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 748.25 | 746.22 | 743.75 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 734.95 | 741.56 | 742.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 730.60 | 739.37 | 741.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 737.30 | 737.14 | 739.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 737.30 | 737.14 | 739.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 737.30 | 737.14 | 739.74 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 765.20 | 745.77 | 743.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 15:15:00 | 774.00 | 760.38 | 751.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 11:15:00 | 770.00 | 770.22 | 764.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:00:00 | 770.00 | 770.22 | 764.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 760.05 | 770.19 | 766.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 757.15 | 770.19 | 766.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 758.30 | 767.81 | 766.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:45:00 | 756.75 | 767.81 | 766.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 768.60 | 769.49 | 767.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 765.90 | 769.49 | 767.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 768.15 | 769.22 | 767.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 768.60 | 769.22 | 767.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 774.45 | 770.27 | 768.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 780.25 | 772.78 | 770.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:00:00 | 778.20 | 772.78 | 770.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:15:00 | 784.85 | 771.64 | 770.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 765.10 | 803.86 | 802.98 | SL hit (close<static) qty=1.00 sl=768.15 alert=retest2 |

### Cycle 125 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 766.85 | 796.46 | 799.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 757.85 | 768.67 | 775.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 764.10 | 762.38 | 768.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:30:00 | 762.20 | 762.38 | 768.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 757.15 | 736.71 | 741.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 757.15 | 736.71 | 741.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 745.00 | 738.37 | 741.46 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 750.95 | 744.73 | 743.94 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 15:15:00 | 738.00 | 743.26 | 743.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 737.60 | 741.35 | 742.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 13:15:00 | 740.00 | 739.27 | 741.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 738.30 | 739.27 | 741.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 739.85 | 739.38 | 741.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:45:00 | 741.00 | 739.38 | 741.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 738.40 | 739.19 | 740.83 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-03 10:30:00 | 655.75 | 2024-06-06 09:15:00 | 669.30 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-06-03 12:00:00 | 655.75 | 2024-06-06 09:15:00 | 669.30 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-06-03 12:45:00 | 655.05 | 2024-06-06 09:15:00 | 669.30 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-06-03 14:00:00 | 655.25 | 2024-06-06 09:15:00 | 669.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-06-04 09:15:00 | 638.00 | 2024-06-06 09:15:00 | 669.30 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2024-06-11 10:15:00 | 735.65 | 2024-06-18 10:15:00 | 724.55 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-06-24 10:15:00 | 812.95 | 2024-06-25 13:15:00 | 776.00 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2024-06-27 12:45:00 | 773.70 | 2024-07-01 14:15:00 | 777.70 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-06-27 14:45:00 | 774.00 | 2024-07-01 14:15:00 | 777.70 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-06-28 09:15:00 | 767.80 | 2024-07-01 14:15:00 | 777.70 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-07-01 12:15:00 | 776.70 | 2024-07-01 14:15:00 | 777.70 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-07-05 10:00:00 | 797.55 | 2024-07-09 09:15:00 | 877.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-05 11:15:00 | 796.95 | 2024-07-09 09:15:00 | 876.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-26 13:30:00 | 880.55 | 2024-08-05 09:15:00 | 836.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-29 09:30:00 | 881.55 | 2024-08-05 09:15:00 | 837.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-29 10:30:00 | 881.70 | 2024-08-05 09:15:00 | 837.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 10:00:00 | 876.05 | 2024-08-05 09:15:00 | 832.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 13:00:00 | 877.40 | 2024-08-05 09:15:00 | 833.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-30 15:00:00 | 876.20 | 2024-08-05 09:15:00 | 832.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 09:15:00 | 866.90 | 2024-08-05 09:15:00 | 823.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-26 13:30:00 | 880.55 | 2024-08-06 09:15:00 | 829.00 | STOP_HIT | 0.50 | 5.85% |
| SELL | retest2 | 2024-07-29 09:30:00 | 881.55 | 2024-08-06 09:15:00 | 829.00 | STOP_HIT | 0.50 | 5.96% |
| SELL | retest2 | 2024-07-29 10:30:00 | 881.70 | 2024-08-06 09:15:00 | 829.00 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2024-07-30 10:00:00 | 876.05 | 2024-08-06 09:15:00 | 829.00 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2024-07-30 13:00:00 | 877.40 | 2024-08-06 09:15:00 | 829.00 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2024-07-30 15:00:00 | 876.20 | 2024-08-06 09:15:00 | 829.00 | STOP_HIT | 0.50 | 5.39% |
| SELL | retest2 | 2024-07-31 09:15:00 | 866.90 | 2024-08-06 09:15:00 | 829.00 | STOP_HIT | 0.50 | 4.37% |
| BUY | retest2 | 2024-08-12 13:30:00 | 827.75 | 2024-08-14 15:15:00 | 828.80 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2024-08-30 12:15:00 | 860.00 | 2024-09-02 09:15:00 | 878.90 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-08-30 15:00:00 | 858.75 | 2024-09-02 09:15:00 | 878.90 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-09-02 13:30:00 | 859.45 | 2024-09-03 09:15:00 | 881.65 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-09-10 10:15:00 | 858.55 | 2024-09-16 11:15:00 | 867.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-10 11:00:00 | 858.25 | 2024-09-16 11:15:00 | 867.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-09-10 11:45:00 | 854.55 | 2024-09-16 11:15:00 | 867.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-09-11 14:45:00 | 855.00 | 2024-09-16 11:15:00 | 867.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-09-19 15:15:00 | 890.00 | 2024-09-23 09:15:00 | 861.85 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-09-20 12:30:00 | 882.00 | 2024-09-23 09:15:00 | 861.85 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-09-23 15:15:00 | 865.00 | 2024-09-25 14:15:00 | 879.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-09-25 12:00:00 | 865.40 | 2024-09-25 14:15:00 | 879.80 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-10-16 10:15:00 | 771.45 | 2024-10-16 13:15:00 | 787.50 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-10-16 12:45:00 | 773.95 | 2024-10-16 13:15:00 | 787.50 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-10-24 11:00:00 | 761.25 | 2024-10-31 15:15:00 | 757.00 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-11-06 10:15:00 | 774.35 | 2024-11-08 09:15:00 | 753.75 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-11-06 11:00:00 | 775.80 | 2024-11-08 09:15:00 | 753.75 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-11-07 10:45:00 | 773.80 | 2024-11-08 09:15:00 | 753.75 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-11-07 11:15:00 | 773.65 | 2024-11-08 09:15:00 | 753.75 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-11-19 11:30:00 | 699.05 | 2024-11-25 09:15:00 | 698.45 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-11-27 11:15:00 | 709.00 | 2024-12-09 09:15:00 | 779.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-23 09:15:00 | 694.95 | 2024-12-30 13:15:00 | 663.95 | PARTIAL | 0.50 | 4.46% |
| SELL | retest2 | 2024-12-23 11:45:00 | 698.90 | 2024-12-30 14:15:00 | 660.20 | PARTIAL | 0.50 | 5.54% |
| SELL | retest2 | 2024-12-23 09:15:00 | 694.95 | 2024-12-31 13:15:00 | 661.45 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest2 | 2024-12-23 11:45:00 | 698.90 | 2024-12-31 13:15:00 | 661.45 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2025-01-29 11:15:00 | 595.05 | 2025-01-29 12:15:00 | 597.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-01-29 11:45:00 | 595.05 | 2025-01-29 12:15:00 | 597.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-01-29 12:15:00 | 594.55 | 2025-01-29 12:15:00 | 597.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-01-31 12:00:00 | 611.65 | 2025-02-03 11:15:00 | 612.55 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest1 | 2025-01-31 13:30:00 | 610.90 | 2025-02-03 11:15:00 | 612.55 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-02-03 10:30:00 | 626.45 | 2025-02-03 12:15:00 | 598.75 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2025-03-04 12:00:00 | 545.65 | 2025-03-04 14:15:00 | 551.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-03-04 12:45:00 | 545.10 | 2025-03-04 14:15:00 | 551.60 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-03-18 15:15:00 | 603.00 | 2025-03-25 09:15:00 | 663.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-19 12:15:00 | 603.75 | 2025-03-25 09:15:00 | 664.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-01 09:15:00 | 624.25 | 2025-04-04 15:15:00 | 593.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 10:15:00 | 619.70 | 2025-04-04 15:15:00 | 593.27 | PARTIAL | 0.50 | 4.26% |
| SELL | retest2 | 2025-04-01 09:15:00 | 624.25 | 2025-04-07 09:15:00 | 561.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 10:15:00 | 619.70 | 2025-04-07 09:15:00 | 557.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-02 12:00:00 | 624.50 | 2025-04-07 09:15:00 | 562.05 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-03 09:30:00 | 620.35 | 2025-04-07 09:15:00 | 558.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 615.80 | 2025-04-07 09:15:00 | 554.22 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-23 13:15:00 | 609.00 | 2025-04-23 13:15:00 | 624.30 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-05-08 09:45:00 | 658.30 | 2025-05-12 09:15:00 | 724.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 09:45:00 | 660.05 | 2025-05-16 13:15:00 | 692.55 | STOP_HIT | 1.00 | 4.92% |
| BUY | retest2 | 2025-05-22 15:00:00 | 713.45 | 2025-05-23 12:15:00 | 701.05 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-05-23 09:15:00 | 710.85 | 2025-05-23 12:15:00 | 701.05 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-05-23 10:45:00 | 711.00 | 2025-05-23 12:15:00 | 701.05 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-05-23 12:00:00 | 713.60 | 2025-05-23 12:15:00 | 701.05 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-06-04 11:15:00 | 724.00 | 2025-06-04 12:15:00 | 718.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-17 10:15:00 | 775.00 | 2025-06-20 11:15:00 | 776.90 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-07-17 10:00:00 | 820.05 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-17 10:30:00 | 821.95 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-18 11:30:00 | 821.25 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-18 12:15:00 | 821.50 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-18 14:30:00 | 826.15 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-07-30 09:15:00 | 792.15 | 2025-08-01 09:15:00 | 808.05 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-31 09:30:00 | 790.00 | 2025-08-01 09:15:00 | 808.05 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-07-31 14:30:00 | 788.00 | 2025-08-01 09:15:00 | 808.05 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-08-01 09:30:00 | 792.05 | 2025-08-01 10:15:00 | 817.00 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-08-05 09:45:00 | 822.55 | 2025-08-06 10:15:00 | 805.05 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-08-05 10:30:00 | 822.10 | 2025-08-06 10:15:00 | 805.05 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-03 10:15:00 | 779.05 | 2025-09-03 15:15:00 | 773.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-09-11 14:45:00 | 791.20 | 2025-09-26 09:15:00 | 870.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-06 11:15:00 | 837.90 | 2025-10-08 09:15:00 | 796.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-06 11:15:00 | 837.90 | 2025-10-09 13:15:00 | 793.35 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2025-11-14 13:30:00 | 651.00 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-11-14 14:45:00 | 644.00 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-11-17 09:30:00 | 649.30 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-11-17 12:00:00 | 650.25 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-18 09:15:00 | 649.25 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-12-02 15:15:00 | 636.00 | 2025-12-04 09:15:00 | 640.70 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-03 10:15:00 | 635.90 | 2025-12-04 09:15:00 | 640.70 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-03 12:30:00 | 636.25 | 2025-12-04 09:15:00 | 640.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-12-04 10:45:00 | 636.35 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-04 12:15:00 | 633.05 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-12-04 13:30:00 | 634.55 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-05 09:15:00 | 630.80 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-12-09 12:00:00 | 633.65 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-12-11 10:15:00 | 660.05 | 2025-12-12 15:15:00 | 650.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-12-11 11:15:00 | 660.00 | 2025-12-12 15:15:00 | 650.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-12-11 11:45:00 | 663.90 | 2025-12-12 15:15:00 | 650.00 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-12-12 09:15:00 | 660.20 | 2025-12-12 15:15:00 | 650.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-12-18 09:15:00 | 637.60 | 2025-12-22 10:15:00 | 648.30 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-12-24 09:15:00 | 656.20 | 2025-12-26 11:15:00 | 639.65 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-01-02 10:30:00 | 648.00 | 2026-01-05 10:15:00 | 642.55 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-02 13:15:00 | 648.45 | 2026-01-05 10:15:00 | 642.55 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-02 14:30:00 | 647.70 | 2026-01-05 10:15:00 | 642.55 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-06 09:15:00 | 635.20 | 2026-01-06 14:15:00 | 603.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 635.20 | 2026-01-09 15:15:00 | 571.68 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-29 13:00:00 | 523.55 | 2026-01-30 15:15:00 | 575.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 09:30:00 | 547.00 | 2026-02-03 09:15:00 | 601.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-17 12:15:00 | 630.70 | 2026-02-18 09:15:00 | 632.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-02-17 15:00:00 | 630.45 | 2026-02-18 09:15:00 | 632.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-02-18 09:15:00 | 630.40 | 2026-02-18 09:15:00 | 632.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-02-20 11:30:00 | 672.00 | 2026-02-25 11:15:00 | 646.80 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2026-02-23 10:45:00 | 658.00 | 2026-02-25 11:15:00 | 646.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-02-23 12:30:00 | 654.60 | 2026-02-25 11:15:00 | 646.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-24 10:15:00 | 661.00 | 2026-02-25 11:15:00 | 646.80 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-02-26 11:30:00 | 644.40 | 2026-02-26 13:15:00 | 651.60 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-03-12 10:15:00 | 702.10 | 2026-03-13 09:15:00 | 687.40 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-03-12 11:45:00 | 695.30 | 2026-03-13 09:15:00 | 687.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-03-19 09:15:00 | 662.30 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-03-19 10:15:00 | 664.15 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-20 12:15:00 | 664.00 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-20 13:00:00 | 663.85 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-03-20 15:15:00 | 653.90 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2026-03-24 10:00:00 | 652.50 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2026-04-02 13:45:00 | 731.25 | 2026-04-06 09:15:00 | 804.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 15:00:00 | 734.95 | 2026-04-07 12:15:00 | 726.55 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-04-07 10:30:00 | 737.15 | 2026-04-07 12:15:00 | 726.55 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-04-22 09:30:00 | 780.25 | 2026-04-27 09:15:00 | 765.10 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-04-22 10:00:00 | 778.20 | 2026-04-27 09:15:00 | 765.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-04-22 13:15:00 | 784.85 | 2026-04-27 09:15:00 | 765.10 | STOP_HIT | 1.00 | -2.52% |
