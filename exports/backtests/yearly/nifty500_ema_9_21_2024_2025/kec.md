# Kec International Ltd. (KEC)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 597.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 149 |
| ALERT1 | 106 |
| ALERT2 | 107 |
| ALERT2_SKIP | 54 |
| ALERT3 | 255 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 124 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 119 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 104
- **Target hits / Stop hits / Partials:** 9 / 119 / 9
- **Avg / median % per leg:** -0.19% / -0.90%
- **Sum % (uncompounded):** -25.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 12 | 30.0% | 7 | 33 | 0 | 0.71% | 28.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.03% | -4.0% |
| BUY @ 3rd Alert (retest2) | 39 | 12 | 30.8% | 7 | 32 | 0 | 0.83% | 32.3% |
| SELL (all) | 97 | 21 | 21.6% | 2 | 86 | 9 | -0.56% | -53.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.56% | -16.7% |
| SELL @ 3rd Alert (retest2) | 94 | 21 | 22.3% | 2 | 83 | 9 | -0.40% | -37.2% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.18% | -20.7% |
| retest2 (combined) | 133 | 33 | 24.8% | 9 | 115 | 9 | -0.04% | -5.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 746.45 | 725.06 | 724.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 786.80 | 737.41 | 730.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 795.50 | 795.65 | 786.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 795.50 | 795.65 | 786.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 799.75 | 796.64 | 788.92 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 782.10 | 790.88 | 790.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 15:15:00 | 780.00 | 784.02 | 786.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 11:15:00 | 782.75 | 782.48 | 785.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 11:15:00 | 782.75 | 782.48 | 785.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 782.75 | 782.48 | 785.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:45:00 | 786.55 | 782.48 | 785.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 747.55 | 756.34 | 761.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:15:00 | 746.40 | 756.34 | 761.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:00:00 | 746.05 | 754.29 | 759.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 745.10 | 752.43 | 757.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 778.20 | 753.62 | 756.48 | SL hit (close>static) qty=1.00 sl=761.35 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 776.45 | 761.59 | 759.81 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 705.05 | 752.07 | 757.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 15:15:00 | 694.00 | 724.44 | 741.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 720.00 | 719.82 | 731.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 763.15 | 728.51 | 733.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 763.15 | 728.51 | 733.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 764.10 | 728.51 | 733.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 777.50 | 738.31 | 737.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 809.45 | 776.87 | 765.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 903.60 | 923.32 | 906.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 903.60 | 923.32 | 906.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 903.60 | 923.32 | 906.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 901.15 | 923.32 | 906.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 897.15 | 918.08 | 905.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 895.85 | 918.08 | 905.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 900.20 | 914.51 | 904.78 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 876.40 | 897.59 | 899.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 14:15:00 | 870.60 | 881.33 | 889.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 15:15:00 | 870.00 | 869.86 | 877.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 09:15:00 | 866.15 | 869.86 | 877.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 861.55 | 868.20 | 876.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:45:00 | 855.00 | 864.95 | 869.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:30:00 | 856.50 | 862.54 | 867.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 911.60 | 872.92 | 870.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 911.60 | 872.92 | 870.06 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 879.00 | 884.29 | 884.35 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 14:15:00 | 904.00 | 887.42 | 885.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 952.50 | 902.93 | 892.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 12:15:00 | 925.20 | 926.51 | 915.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 13:00:00 | 925.20 | 926.51 | 915.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 923.75 | 925.23 | 917.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 912.75 | 925.23 | 917.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 910.80 | 922.34 | 916.95 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 12:15:00 | 904.00 | 913.63 | 913.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 13:15:00 | 900.30 | 910.97 | 912.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 15:15:00 | 900.75 | 900.34 | 904.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:15:00 | 900.95 | 900.34 | 904.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 898.95 | 900.07 | 904.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 887.50 | 903.18 | 903.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 12:15:00 | 885.40 | 879.78 | 879.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 12:15:00 | 885.40 | 879.78 | 879.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 894.35 | 885.12 | 882.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 888.20 | 888.96 | 885.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:30:00 | 888.00 | 888.96 | 885.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 888.05 | 888.38 | 885.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 885.50 | 888.38 | 885.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 884.10 | 887.53 | 885.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 919.60 | 887.53 | 885.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 15:15:00 | 892.00 | 900.06 | 894.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 874.90 | 893.74 | 892.65 | SL hit (close<static) qty=1.00 sl=884.10 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 873.75 | 889.74 | 890.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 865.80 | 879.86 | 885.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 877.95 | 876.93 | 881.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 11:15:00 | 877.95 | 876.93 | 881.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 877.95 | 876.93 | 881.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 879.60 | 876.93 | 881.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 880.00 | 876.73 | 880.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 876.15 | 876.73 | 880.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 871.65 | 875.71 | 879.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 844.70 | 874.97 | 878.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 12:15:00 | 886.10 | 877.35 | 877.74 | SL hit (close>static) qty=1.00 sl=884.45 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 885.50 | 878.88 | 878.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 893.00 | 881.71 | 879.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 876.15 | 884.11 | 882.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 15:15:00 | 876.15 | 884.11 | 882.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 876.15 | 884.11 | 882.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 879.70 | 884.11 | 882.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 880.70 | 883.43 | 882.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:30:00 | 876.65 | 883.43 | 882.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 884.80 | 883.70 | 882.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:45:00 | 886.00 | 883.70 | 882.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 883.40 | 883.64 | 882.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:30:00 | 884.00 | 883.64 | 882.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 887.90 | 884.49 | 883.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:45:00 | 885.75 | 884.49 | 883.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 877.55 | 883.10 | 882.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:45:00 | 878.75 | 883.10 | 882.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 880.25 | 882.53 | 882.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:30:00 | 876.80 | 882.53 | 882.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 15:15:00 | 880.00 | 882.03 | 882.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 11:15:00 | 877.95 | 881.13 | 881.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 11:15:00 | 883.95 | 877.73 | 879.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 11:15:00 | 883.95 | 877.73 | 879.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 883.95 | 877.73 | 879.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:00:00 | 883.95 | 877.73 | 879.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 888.00 | 879.79 | 879.92 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 887.00 | 881.23 | 880.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 922.40 | 890.70 | 885.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 11:15:00 | 917.60 | 917.74 | 906.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 12:00:00 | 917.60 | 917.74 | 906.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 907.85 | 915.76 | 906.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:00:00 | 907.85 | 915.76 | 906.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 906.10 | 913.83 | 906.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:00:00 | 906.10 | 913.83 | 906.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 906.70 | 912.40 | 906.41 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 889.30 | 901.76 | 902.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 888.25 | 899.06 | 901.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 839.35 | 834.10 | 846.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 09:45:00 | 841.00 | 834.10 | 846.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 848.00 | 836.88 | 846.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 846.95 | 836.88 | 846.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 850.80 | 839.66 | 847.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 851.30 | 839.66 | 847.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 849.60 | 841.65 | 847.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 852.60 | 841.65 | 847.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 856.00 | 844.52 | 848.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 856.00 | 844.52 | 848.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 849.75 | 845.57 | 848.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:45:00 | 855.45 | 845.57 | 848.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 849.75 | 847.91 | 849.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:00:00 | 849.75 | 847.91 | 849.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 856.05 | 849.54 | 849.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:30:00 | 856.20 | 849.54 | 849.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 857.95 | 851.22 | 850.42 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 845.65 | 850.35 | 850.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 12:15:00 | 840.00 | 848.28 | 849.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 10:15:00 | 833.25 | 832.97 | 837.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 11:00:00 | 833.25 | 832.97 | 837.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 833.20 | 833.42 | 837.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 14:30:00 | 824.45 | 832.05 | 835.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:30:00 | 827.10 | 818.97 | 821.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 11:15:00 | 830.70 | 821.45 | 822.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 839.40 | 826.23 | 824.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 839.40 | 826.23 | 824.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 14:15:00 | 848.20 | 841.90 | 836.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 839.10 | 842.08 | 837.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 839.10 | 842.08 | 837.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 839.10 | 842.08 | 837.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:45:00 | 838.90 | 842.08 | 837.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 831.55 | 839.97 | 837.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 831.55 | 839.97 | 837.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 826.55 | 837.29 | 836.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:00:00 | 826.55 | 837.29 | 836.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 12:15:00 | 828.75 | 835.58 | 835.59 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 12:15:00 | 841.50 | 835.76 | 835.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 870.45 | 844.37 | 839.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 14:15:00 | 855.45 | 855.64 | 847.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 15:00:00 | 855.45 | 855.64 | 847.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 855.15 | 854.87 | 848.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 851.10 | 854.87 | 848.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 860.65 | 860.52 | 855.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 856.20 | 860.52 | 855.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 968.20 | 987.35 | 974.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 967.45 | 987.35 | 974.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 982.85 | 986.45 | 975.70 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 961.20 | 975.94 | 976.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 10:15:00 | 957.80 | 972.31 | 974.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 982.60 | 965.53 | 968.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 982.60 | 965.53 | 968.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 982.60 | 965.53 | 968.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 982.60 | 965.53 | 968.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 975.30 | 967.49 | 969.43 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 984.20 | 970.83 | 970.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 12:15:00 | 989.85 | 974.63 | 972.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 09:15:00 | 982.00 | 982.95 | 977.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:00:00 | 982.00 | 982.95 | 977.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 987.55 | 983.36 | 978.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 15:15:00 | 996.00 | 986.57 | 981.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 977.45 | 986.25 | 982.43 | SL hit (close<static) qty=1.00 sl=978.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 965.30 | 978.39 | 979.72 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 11:15:00 | 1005.65 | 983.66 | 981.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 12:15:00 | 1015.00 | 989.93 | 984.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 992.50 | 997.95 | 990.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 992.50 | 997.95 | 990.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 992.50 | 997.95 | 990.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 991.95 | 997.95 | 990.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 980.75 | 994.41 | 990.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 977.10 | 994.41 | 990.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 990.50 | 993.63 | 990.06 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 09:15:00 | 943.45 | 981.23 | 985.25 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 974.20 | 961.02 | 960.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 986.80 | 966.17 | 963.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 15:15:00 | 1019.80 | 1019.85 | 1006.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:15:00 | 1013.70 | 1019.85 | 1006.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 1010.50 | 1016.20 | 1007.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:00:00 | 1010.50 | 1016.20 | 1007.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 1007.70 | 1014.50 | 1007.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:00:00 | 1007.70 | 1014.50 | 1007.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1022.10 | 1016.02 | 1008.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 14:45:00 | 1026.95 | 1017.82 | 1010.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1040.00 | 1019.35 | 1011.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:15:00 | 1037.90 | 1044.00 | 1038.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 12:15:00 | 1018.00 | 1035.01 | 1035.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 1018.00 | 1035.01 | 1035.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 1015.00 | 1026.37 | 1029.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 13:15:00 | 1026.50 | 1026.39 | 1029.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 14:00:00 | 1026.50 | 1026.39 | 1029.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1035.45 | 1028.21 | 1030.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 1035.45 | 1028.21 | 1030.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1032.05 | 1028.97 | 1030.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1016.30 | 1028.97 | 1030.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 10:15:00 | 1019.90 | 1008.60 | 1008.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1019.90 | 1008.60 | 1008.14 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 1007.05 | 1007.91 | 1007.91 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 15:15:00 | 1010.00 | 1008.33 | 1008.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1020.00 | 1010.66 | 1009.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 1016.20 | 1016.83 | 1013.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 1011.85 | 1016.83 | 1013.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1010.00 | 1015.47 | 1013.30 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 998.80 | 1010.52 | 1011.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 12:15:00 | 996.60 | 1007.74 | 1009.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 1008.50 | 1001.22 | 1004.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 14:15:00 | 1008.50 | 1001.22 | 1004.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 1008.50 | 1001.22 | 1004.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 15:00:00 | 1008.50 | 1001.22 | 1004.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 1011.00 | 1003.18 | 1004.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 1025.00 | 1003.18 | 1004.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 1025.20 | 1007.58 | 1006.53 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 999.55 | 1010.36 | 1011.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 996.90 | 1006.00 | 1009.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 901.25 | 898.61 | 912.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 11:15:00 | 901.55 | 898.61 | 912.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 909.20 | 903.72 | 909.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:45:00 | 906.10 | 904.59 | 909.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:15:00 | 904.60 | 904.59 | 909.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:00:00 | 904.90 | 904.65 | 908.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:15:00 | 903.90 | 905.02 | 908.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 919.90 | 908.00 | 909.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 919.90 | 908.00 | 909.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 924.45 | 911.29 | 910.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 924.45 | 911.29 | 910.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 934.45 | 918.11 | 914.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 963.65 | 978.70 | 960.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 963.65 | 978.70 | 960.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 963.65 | 978.70 | 960.43 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 13:15:00 | 956.20 | 961.33 | 961.82 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 972.60 | 960.25 | 960.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 976.50 | 963.50 | 961.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 1026.65 | 1033.96 | 1009.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 10:00:00 | 1026.65 | 1033.96 | 1009.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 1025.55 | 1028.61 | 1015.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 1025.55 | 1028.61 | 1015.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 1021.00 | 1027.09 | 1016.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 1019.95 | 1027.09 | 1016.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 998.90 | 1021.45 | 1014.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:00:00 | 998.90 | 1021.45 | 1014.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1000.15 | 1017.19 | 1013.46 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 997.10 | 1010.29 | 1010.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 971.90 | 995.99 | 1001.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 983.70 | 978.87 | 988.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 983.70 | 978.87 | 988.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 983.70 | 978.87 | 988.45 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 14:15:00 | 1009.40 | 993.75 | 992.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 09:15:00 | 1019.55 | 1001.18 | 996.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 13:15:00 | 1006.20 | 1007.05 | 1001.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-18 14:00:00 | 1006.20 | 1007.05 | 1001.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1007.25 | 1007.09 | 1001.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:30:00 | 1004.05 | 1007.09 | 1001.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 1004.10 | 1006.49 | 1001.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 1014.05 | 1006.49 | 1001.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1008.25 | 1006.84 | 1002.55 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 984.25 | 998.96 | 1000.94 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1007.70 | 999.77 | 998.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 1027.00 | 1005.21 | 1001.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 1004.55 | 1021.63 | 1013.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 1004.55 | 1021.63 | 1013.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1004.55 | 1021.63 | 1013.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:45:00 | 1006.95 | 1021.63 | 1013.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 1006.00 | 1018.50 | 1012.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 11:15:00 | 1007.10 | 1018.50 | 1012.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:00:00 | 1007.05 | 1016.21 | 1012.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:45:00 | 1008.55 | 1014.32 | 1011.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 13:45:00 | 1009.10 | 1013.30 | 1011.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1033.80 | 1020.46 | 1015.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 1016.45 | 1020.46 | 1015.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 1022.85 | 1023.01 | 1019.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 1071.00 | 1023.01 | 1019.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-03 10:15:00 | 1107.81 | 1069.34 | 1057.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 1192.95 | 1197.50 | 1197.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 1189.90 | 1195.98 | 1197.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 1202.10 | 1176.27 | 1182.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1202.10 | 1176.27 | 1182.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1202.10 | 1176.27 | 1182.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 1203.10 | 1176.27 | 1182.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1189.95 | 1179.00 | 1183.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 1183.95 | 1184.27 | 1184.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:00:00 | 1184.35 | 1183.84 | 1184.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:00:00 | 1181.25 | 1172.73 | 1175.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 15:00:00 | 1183.60 | 1174.90 | 1176.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 1202.00 | 1177.77 | 1177.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 10:15:00 | 1202.00 | 1177.77 | 1177.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 11:15:00 | 1210.00 | 1184.21 | 1180.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 15:15:00 | 1220.00 | 1226.48 | 1210.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 09:15:00 | 1211.35 | 1226.48 | 1210.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1201.95 | 1221.58 | 1209.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 1204.80 | 1221.58 | 1209.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1213.75 | 1220.01 | 1210.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 14:45:00 | 1220.05 | 1217.80 | 1212.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 1181.00 | 1211.11 | 1210.04 | SL hit (close<static) qty=1.00 sl=1196.95 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 10:15:00 | 1185.55 | 1205.99 | 1207.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 1164.90 | 1182.22 | 1192.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 09:15:00 | 1191.70 | 1167.89 | 1175.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 1191.70 | 1167.89 | 1175.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1191.70 | 1167.89 | 1175.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 1180.75 | 1167.89 | 1175.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1181.40 | 1170.59 | 1175.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:30:00 | 1189.05 | 1170.59 | 1175.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 1182.55 | 1175.62 | 1177.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:45:00 | 1193.90 | 1175.62 | 1177.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1202.20 | 1182.95 | 1180.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 1215.00 | 1196.82 | 1189.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 13:15:00 | 1202.00 | 1204.40 | 1196.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 14:00:00 | 1202.00 | 1204.40 | 1196.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 1200.00 | 1202.69 | 1196.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 1204.25 | 1202.69 | 1196.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 11:45:00 | 1203.65 | 1211.07 | 1208.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 14:15:00 | 1185.10 | 1202.93 | 1205.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 1185.10 | 1202.93 | 1205.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 1148.55 | 1189.67 | 1198.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 950.60 | 948.54 | 980.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 09:15:00 | 959.35 | 952.73 | 967.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 959.35 | 952.73 | 967.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 949.85 | 958.26 | 965.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 980.75 | 964.29 | 966.56 | SL hit (close>static) qty=1.00 sl=975.95 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 990.80 | 972.42 | 970.04 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 969.40 | 972.70 | 972.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 15:15:00 | 965.00 | 971.16 | 972.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 14:15:00 | 970.25 | 966.11 | 968.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 14:15:00 | 970.25 | 966.11 | 968.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 970.25 | 966.11 | 968.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:00:00 | 970.25 | 966.11 | 968.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 970.20 | 966.93 | 968.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 973.95 | 966.93 | 968.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 956.30 | 964.80 | 967.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:15:00 | 952.00 | 964.80 | 967.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:00:00 | 954.95 | 960.60 | 964.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:15:00 | 904.40 | 935.27 | 949.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:15:00 | 907.20 | 935.27 | 949.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-23 09:15:00 | 859.46 | 900.82 | 923.05 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 816.95 | 805.91 | 805.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 832.00 | 812.99 | 809.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 803.75 | 811.14 | 808.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 803.75 | 811.14 | 808.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 803.75 | 811.14 | 808.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 803.75 | 811.14 | 808.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 812.10 | 811.33 | 808.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:30:00 | 820.70 | 813.10 | 810.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-01 11:15:00 | 902.77 | 856.96 | 835.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 794.55 | 824.86 | 827.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 768.40 | 796.50 | 810.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 13:15:00 | 792.15 | 787.27 | 800.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 13:45:00 | 790.00 | 787.27 | 800.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 810.00 | 791.82 | 801.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:30:00 | 816.15 | 791.82 | 801.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 813.00 | 796.05 | 802.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 823.25 | 796.05 | 802.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 834.35 | 807.11 | 806.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 14:15:00 | 847.00 | 825.18 | 816.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 826.70 | 828.82 | 819.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:30:00 | 827.90 | 828.82 | 819.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 812.55 | 825.56 | 818.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 812.55 | 825.56 | 818.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 810.00 | 822.45 | 818.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 810.00 | 822.45 | 818.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 828.75 | 818.63 | 817.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:15:00 | 832.05 | 821.29 | 818.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 833.70 | 823.10 | 820.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 806.95 | 819.32 | 819.19 | SL hit (close<static) qty=1.00 sl=809.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 813.20 | 817.93 | 818.57 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 14:15:00 | 825.00 | 819.35 | 819.16 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 796.85 | 815.91 | 817.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 788.85 | 805.84 | 812.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 778.05 | 777.81 | 793.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 778.05 | 777.81 | 793.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 782.65 | 776.74 | 786.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 752.35 | 775.99 | 782.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 714.73 | 739.65 | 759.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-14 14:15:00 | 748.00 | 741.32 | 758.74 | SL hit (close>ema200) qty=0.50 sl=741.32 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 742.20 | 737.21 | 736.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 758.75 | 743.23 | 739.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 751.65 | 758.41 | 751.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 751.65 | 758.41 | 751.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 751.65 | 758.41 | 751.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 751.65 | 758.41 | 751.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 754.90 | 757.71 | 751.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 761.70 | 758.16 | 752.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 13:15:00 | 750.05 | 751.47 | 751.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 750.05 | 751.47 | 751.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 746.30 | 749.96 | 750.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 747.60 | 746.96 | 748.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 13:00:00 | 747.60 | 746.96 | 748.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 685.00 | 674.51 | 685.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:30:00 | 697.90 | 679.67 | 686.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 700.00 | 683.74 | 687.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 700.00 | 683.74 | 687.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 699.80 | 690.56 | 690.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 701.70 | 692.79 | 691.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 15:15:00 | 692.10 | 692.65 | 691.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 15:15:00 | 692.10 | 692.65 | 691.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 692.10 | 692.65 | 691.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 708.90 | 692.65 | 691.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 704.55 | 695.03 | 692.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 12:15:00 | 710.50 | 699.27 | 694.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 12:45:00 | 713.20 | 702.23 | 696.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 12:15:00 | 706.75 | 716.75 | 716.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 706.75 | 716.75 | 716.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 704.00 | 714.20 | 715.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 687.20 | 685.65 | 695.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:45:00 | 687.60 | 685.65 | 695.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 689.70 | 686.46 | 695.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 694.60 | 686.46 | 695.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 698.35 | 689.72 | 695.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:30:00 | 697.30 | 689.72 | 695.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 701.15 | 692.00 | 695.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:30:00 | 698.45 | 692.00 | 695.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 713.30 | 696.26 | 697.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 713.30 | 696.26 | 697.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 684.30 | 694.55 | 696.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 680.00 | 691.18 | 694.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 714.50 | 686.34 | 689.20 | SL hit (close>static) qty=1.00 sl=706.55 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 712.50 | 691.57 | 691.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 15:15:00 | 726.00 | 708.61 | 700.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 15:15:00 | 832.00 | 835.01 | 811.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 09:15:00 | 801.00 | 835.01 | 811.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 798.75 | 827.76 | 810.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 798.55 | 827.76 | 810.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 808.25 | 823.86 | 810.31 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 795.00 | 804.80 | 805.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 11:15:00 | 792.30 | 802.30 | 804.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 771.00 | 768.00 | 776.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:30:00 | 772.55 | 768.00 | 776.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 783.80 | 771.16 | 777.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 783.80 | 771.16 | 777.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 804.00 | 777.73 | 779.60 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 803.50 | 782.88 | 781.77 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 770.45 | 783.64 | 783.87 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 788.25 | 782.76 | 782.61 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 767.55 | 780.88 | 781.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 09:15:00 | 764.35 | 770.11 | 774.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 669.95 | 664.17 | 687.25 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 13:45:00 | 664.55 | 666.54 | 681.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 14:45:00 | 664.90 | 666.23 | 679.78 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 09:15:00 | 657.20 | 666.39 | 678.62 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 660.00 | 660.20 | 668.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 699.00 | 667.88 | 667.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 65 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 699.00 | 667.88 | 667.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 717.00 | 677.71 | 672.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 713.00 | 718.96 | 707.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 713.00 | 718.96 | 707.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 711.40 | 714.30 | 709.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 719.60 | 714.30 | 709.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 715.70 | 732.50 | 733.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 715.70 | 732.50 | 733.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 708.85 | 718.36 | 722.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 712.05 | 707.76 | 714.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 712.05 | 707.76 | 714.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 712.05 | 707.76 | 714.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 712.05 | 707.76 | 714.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 705.15 | 707.24 | 713.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:45:00 | 699.90 | 706.28 | 712.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 702.65 | 704.97 | 710.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:15:00 | 703.00 | 704.68 | 709.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:00:00 | 703.10 | 700.97 | 704.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 710.20 | 702.81 | 704.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:45:00 | 710.50 | 702.81 | 704.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 713.35 | 704.92 | 705.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 713.35 | 704.92 | 705.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 702.65 | 699.04 | 702.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 702.70 | 699.04 | 702.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 708.65 | 700.96 | 702.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:45:00 | 704.50 | 700.96 | 702.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 715.75 | 703.92 | 703.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 715.75 | 703.92 | 703.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 12:15:00 | 718.60 | 706.85 | 705.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 714.00 | 717.57 | 713.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 12:15:00 | 714.00 | 717.57 | 713.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 714.00 | 717.57 | 713.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 714.00 | 717.57 | 713.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 706.80 | 715.42 | 712.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 706.80 | 715.42 | 712.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 701.25 | 712.58 | 711.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 701.70 | 712.58 | 711.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 698.40 | 709.75 | 710.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 691.55 | 706.11 | 708.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 722.80 | 702.14 | 703.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 722.80 | 702.14 | 703.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 722.80 | 702.14 | 703.83 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 723.00 | 706.31 | 705.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 732.00 | 718.26 | 712.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 735.00 | 735.13 | 724.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:30:00 | 734.60 | 735.13 | 724.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 800.25 | 810.26 | 797.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 802.00 | 810.26 | 797.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 797.20 | 806.55 | 797.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 797.20 | 806.55 | 797.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 798.60 | 804.96 | 797.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 798.60 | 804.96 | 797.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 793.40 | 802.65 | 797.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 793.40 | 802.65 | 797.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 781.35 | 798.39 | 795.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 781.35 | 798.39 | 795.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 784.00 | 795.51 | 794.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 790.65 | 795.51 | 794.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 796.45 | 797.07 | 795.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 796.45 | 797.07 | 795.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 799.40 | 797.53 | 796.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 803.45 | 797.91 | 796.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-27 09:15:00 | 883.80 | 857.14 | 835.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 841.30 | 857.78 | 859.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 831.45 | 845.52 | 852.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 844.45 | 842.95 | 849.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 844.45 | 842.95 | 849.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 844.45 | 842.95 | 849.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 843.85 | 842.95 | 849.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 852.60 | 844.54 | 849.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 852.60 | 844.54 | 849.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 858.25 | 847.28 | 850.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 857.05 | 847.28 | 850.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 863.10 | 854.21 | 853.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 13:15:00 | 870.00 | 860.10 | 856.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 11:15:00 | 874.85 | 875.87 | 869.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 11:30:00 | 874.45 | 875.87 | 869.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 868.35 | 873.65 | 870.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:30:00 | 876.20 | 873.96 | 871.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:00:00 | 875.20 | 873.96 | 871.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:30:00 | 874.50 | 873.53 | 871.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 879.50 | 876.99 | 873.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 924.95 | 916.00 | 907.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 888.15 | 905.53 | 907.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 888.15 | 905.53 | 907.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 875.70 | 896.78 | 903.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 879.25 | 878.97 | 887.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 879.25 | 878.97 | 887.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 879.25 | 878.97 | 887.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:30:00 | 879.95 | 878.97 | 887.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 901.20 | 883.42 | 888.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 902.40 | 883.42 | 888.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 899.10 | 886.55 | 889.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 12:45:00 | 895.65 | 888.50 | 890.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 896.00 | 890.54 | 890.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 904.00 | 893.23 | 892.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 904.00 | 893.23 | 892.02 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 884.50 | 890.93 | 891.42 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 900.75 | 892.33 | 891.78 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 887.60 | 892.02 | 892.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 877.80 | 889.18 | 891.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 15:15:00 | 889.65 | 888.01 | 890.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 15:15:00 | 889.65 | 888.01 | 890.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 889.65 | 888.01 | 890.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 883.20 | 888.01 | 890.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 886.70 | 887.75 | 889.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:15:00 | 890.95 | 887.75 | 889.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 888.95 | 887.99 | 889.69 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 896.50 | 890.78 | 890.56 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 886.40 | 889.77 | 890.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 14:15:00 | 883.30 | 887.06 | 888.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 890.95 | 887.40 | 888.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 890.95 | 887.40 | 888.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 890.95 | 887.40 | 888.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 10:45:00 | 888.00 | 888.93 | 889.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 894.35 | 890.01 | 889.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 894.35 | 890.01 | 889.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 898.15 | 891.64 | 890.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 890.55 | 891.42 | 890.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 890.55 | 891.42 | 890.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 890.55 | 891.42 | 890.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 890.55 | 891.42 | 890.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 888.60 | 890.86 | 890.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 888.60 | 890.86 | 890.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 891.90 | 891.07 | 890.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 932.65 | 891.07 | 890.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 908.80 | 919.15 | 919.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 908.80 | 919.15 | 919.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 903.25 | 910.05 | 913.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 900.35 | 898.73 | 901.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 900.35 | 898.73 | 901.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 893.10 | 897.73 | 900.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:00:00 | 889.30 | 896.05 | 899.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 891.05 | 889.47 | 894.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 13:15:00 | 898.55 | 892.03 | 891.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 898.55 | 892.03 | 891.35 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 880.55 | 891.54 | 892.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 876.80 | 888.59 | 890.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 878.70 | 875.85 | 882.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 878.70 | 875.85 | 882.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 877.45 | 873.27 | 877.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:00:00 | 872.20 | 873.06 | 876.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:45:00 | 873.40 | 876.10 | 877.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 11:15:00 | 882.00 | 878.50 | 878.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 882.00 | 878.50 | 878.14 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 875.80 | 879.08 | 879.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 869.10 | 877.08 | 878.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 875.65 | 873.98 | 876.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 875.65 | 873.98 | 876.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 875.65 | 873.98 | 876.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 875.65 | 873.98 | 876.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 873.00 | 873.78 | 875.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 877.05 | 873.78 | 875.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 878.50 | 874.73 | 876.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:15:00 | 880.75 | 874.73 | 876.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 877.40 | 875.26 | 876.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 876.55 | 875.26 | 876.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:00:00 | 876.40 | 875.42 | 876.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 876.30 | 875.34 | 876.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:00:00 | 877.00 | 875.53 | 876.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 876.40 | 875.70 | 876.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 878.00 | 875.70 | 876.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 879.50 | 876.46 | 876.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 879.50 | 876.46 | 876.35 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 869.95 | 875.64 | 876.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 867.25 | 873.96 | 875.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 862.00 | 861.23 | 866.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 878.95 | 861.23 | 866.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 876.40 | 864.27 | 867.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 878.00 | 864.27 | 867.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 877.10 | 866.83 | 868.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:15:00 | 874.10 | 866.83 | 868.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 12:15:00 | 873.30 | 869.66 | 869.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 873.30 | 869.66 | 869.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 13:15:00 | 878.80 | 871.49 | 870.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 864.40 | 871.75 | 870.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 864.40 | 871.75 | 870.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 864.40 | 871.75 | 870.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 866.00 | 871.75 | 870.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 862.05 | 869.81 | 870.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 859.00 | 864.34 | 866.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 870.70 | 865.62 | 867.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 870.70 | 865.62 | 867.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 870.70 | 865.62 | 867.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 870.70 | 865.62 | 867.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 864.40 | 865.37 | 867.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 862.80 | 865.35 | 866.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:00:00 | 862.85 | 862.07 | 864.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:30:00 | 857.05 | 862.16 | 864.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 13:30:00 | 863.70 | 863.94 | 864.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 858.60 | 862.87 | 863.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 857.60 | 862.68 | 863.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 12:15:00 | 867.65 | 864.35 | 864.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 867.65 | 864.35 | 864.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 873.40 | 866.16 | 865.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 11:15:00 | 868.35 | 870.02 | 867.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 868.35 | 870.02 | 867.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 868.35 | 870.02 | 867.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:45:00 | 866.00 | 870.02 | 867.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 871.15 | 870.25 | 868.06 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 860.00 | 866.38 | 866.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 853.45 | 863.79 | 865.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 832.45 | 832.18 | 842.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:30:00 | 830.90 | 832.18 | 842.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 839.30 | 834.37 | 841.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:30:00 | 833.40 | 834.35 | 839.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:15:00 | 828.45 | 836.93 | 839.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:15:00 | 791.73 | 796.51 | 801.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 796.65 | 795.83 | 800.08 | SL hit (close>ema200) qty=0.50 sl=795.83 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 836.00 | 799.57 | 796.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 844.15 | 808.49 | 800.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 14:15:00 | 827.80 | 831.56 | 822.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 15:00:00 | 827.80 | 831.56 | 822.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 825.75 | 830.14 | 823.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 824.70 | 830.14 | 823.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 823.10 | 827.77 | 824.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 823.10 | 827.77 | 824.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 821.25 | 826.47 | 824.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 821.25 | 826.47 | 824.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 818.60 | 823.16 | 823.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 819.50 | 823.16 | 823.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 819.65 | 822.46 | 822.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 816.10 | 819.82 | 821.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 822.00 | 820.26 | 821.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 822.00 | 820.26 | 821.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 822.00 | 820.26 | 821.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 820.55 | 820.26 | 821.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 819.20 | 820.04 | 821.14 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 13:15:00 | 824.75 | 822.23 | 821.98 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 811.30 | 822.32 | 822.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 804.75 | 811.38 | 815.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 14:15:00 | 812.40 | 810.64 | 814.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 15:00:00 | 812.40 | 810.64 | 814.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 809.00 | 806.18 | 809.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:30:00 | 808.85 | 806.18 | 809.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 803.60 | 805.67 | 809.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 807.15 | 805.67 | 809.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 815.05 | 807.54 | 809.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 814.70 | 807.54 | 809.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 812.65 | 808.56 | 809.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 812.00 | 808.56 | 809.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:00:00 | 811.95 | 809.24 | 810.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:45:00 | 812.00 | 810.17 | 810.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 816.05 | 811.35 | 810.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 816.05 | 811.35 | 810.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 818.35 | 812.75 | 811.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 848.65 | 851.79 | 840.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 848.65 | 851.79 | 840.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 844.15 | 848.98 | 843.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 845.40 | 848.98 | 843.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 861.30 | 851.45 | 845.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 869.00 | 856.95 | 854.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:00:00 | 862.85 | 868.55 | 867.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 858.70 | 866.58 | 866.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 858.70 | 866.58 | 866.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 855.15 | 864.29 | 865.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 863.85 | 862.45 | 864.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 863.85 | 862.45 | 864.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 863.85 | 862.45 | 864.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 864.00 | 862.45 | 864.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 858.85 | 861.73 | 863.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 857.90 | 862.39 | 864.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 880.80 | 866.07 | 865.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 880.80 | 866.07 | 865.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 891.15 | 875.07 | 870.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 878.55 | 880.91 | 876.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 878.55 | 880.91 | 876.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 878.55 | 880.91 | 876.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 893.45 | 880.31 | 877.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:00:00 | 887.25 | 881.70 | 878.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:45:00 | 887.30 | 883.11 | 880.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 875.35 | 880.69 | 879.82 | SL hit (close<static) qty=1.00 sl=876.20 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 873.20 | 879.20 | 879.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 871.40 | 877.64 | 878.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 10:15:00 | 872.15 | 867.98 | 870.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 10:15:00 | 872.15 | 867.98 | 870.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 872.15 | 867.98 | 870.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 873.40 | 867.98 | 870.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 871.50 | 868.68 | 870.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 871.50 | 868.68 | 870.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 872.65 | 869.48 | 870.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 866.75 | 869.22 | 870.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 928.70 | 880.25 | 875.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 928.70 | 880.25 | 875.23 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 877.50 | 882.34 | 882.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 866.10 | 875.68 | 878.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 849.05 | 842.50 | 851.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 849.05 | 842.50 | 851.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 849.05 | 842.50 | 851.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 849.05 | 842.50 | 851.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 853.95 | 844.79 | 851.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:45:00 | 852.20 | 844.79 | 851.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 854.10 | 846.65 | 851.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 854.10 | 846.65 | 851.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 856.50 | 848.62 | 852.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 856.50 | 848.62 | 852.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 858.30 | 850.56 | 852.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 858.30 | 850.56 | 852.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 874.20 | 855.29 | 854.76 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 850.00 | 854.61 | 854.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 11:15:00 | 846.60 | 851.23 | 852.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 14:15:00 | 854.30 | 851.55 | 852.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 854.30 | 851.55 | 852.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 854.30 | 851.55 | 852.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:45:00 | 853.20 | 851.55 | 852.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 853.20 | 851.88 | 852.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 856.05 | 851.88 | 852.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 852.55 | 852.15 | 852.57 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 12:15:00 | 859.80 | 853.67 | 853.19 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 853.00 | 856.74 | 857.20 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 862.70 | 858.21 | 857.71 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 853.60 | 857.07 | 857.27 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 860.35 | 857.49 | 857.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 866.50 | 859.23 | 858.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 13:15:00 | 859.30 | 860.48 | 859.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 13:15:00 | 859.30 | 860.48 | 859.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 859.30 | 860.48 | 859.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 859.30 | 860.48 | 859.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 855.55 | 859.49 | 858.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 855.55 | 859.49 | 858.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 852.00 | 857.99 | 858.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 15:15:00 | 850.50 | 853.39 | 855.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 853.65 | 853.44 | 855.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 853.65 | 853.44 | 855.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 853.65 | 853.44 | 855.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:45:00 | 849.30 | 851.88 | 853.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 12:15:00 | 848.05 | 851.45 | 853.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:15:00 | 848.85 | 847.20 | 849.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 856.40 | 851.16 | 850.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 856.40 | 851.16 | 850.78 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 848.45 | 850.61 | 850.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 846.00 | 848.66 | 849.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 825.80 | 823.97 | 828.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 15:00:00 | 825.80 | 823.97 | 828.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 730.60 | 735.23 | 754.46 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 793.70 | 760.26 | 758.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 802.00 | 789.24 | 782.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 787.65 | 789.94 | 783.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 787.65 | 789.94 | 783.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 787.00 | 790.23 | 786.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 787.00 | 790.23 | 786.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 784.40 | 789.06 | 786.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 784.40 | 789.06 | 786.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 785.95 | 788.44 | 786.63 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 773.95 | 784.57 | 785.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 737.00 | 773.54 | 779.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 707.00 | 700.75 | 708.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 707.00 | 700.75 | 708.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 707.00 | 700.75 | 708.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 707.00 | 700.75 | 708.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 709.00 | 702.40 | 708.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 700.30 | 702.40 | 708.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 702.75 | 702.28 | 705.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:45:00 | 705.85 | 702.07 | 703.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 704.95 | 696.15 | 695.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 704.95 | 696.15 | 695.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 12:15:00 | 709.20 | 700.04 | 697.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 707.40 | 714.77 | 709.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 707.40 | 714.77 | 709.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 707.40 | 714.77 | 709.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 705.20 | 714.77 | 709.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 713.10 | 714.43 | 709.98 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 705.30 | 709.42 | 709.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 14:15:00 | 702.25 | 707.00 | 708.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 15:15:00 | 686.70 | 685.53 | 690.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:15:00 | 688.10 | 685.53 | 690.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 687.95 | 686.02 | 690.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 689.00 | 686.02 | 690.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 685.20 | 685.85 | 690.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 683.45 | 685.85 | 690.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:45:00 | 683.05 | 682.97 | 686.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 13:45:00 | 684.00 | 683.35 | 685.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:15:00 | 683.95 | 683.52 | 685.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 686.85 | 684.26 | 685.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 685.85 | 684.26 | 685.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 684.35 | 684.28 | 685.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 691.50 | 685.93 | 685.97 | SL hit (close>static) qty=1.00 sl=690.65 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 690.65 | 686.88 | 686.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 15:15:00 | 694.45 | 688.93 | 687.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 701.20 | 702.00 | 697.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:00:00 | 701.20 | 702.00 | 697.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 703.40 | 705.01 | 702.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 706.00 | 705.01 | 702.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 702.00 | 704.41 | 702.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 702.00 | 704.41 | 702.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 704.00 | 704.33 | 702.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 704.55 | 704.33 | 702.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 704.65 | 704.28 | 702.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:30:00 | 705.00 | 703.86 | 702.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 700.25 | 703.14 | 702.51 | SL hit (close<static) qty=1.00 sl=700.70 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 15:15:00 | 698.95 | 701.96 | 702.06 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 702.55 | 702.17 | 702.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 705.50 | 703.02 | 702.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 741.25 | 741.63 | 733.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 743.85 | 741.63 | 733.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 742.80 | 740.65 | 736.77 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 733.55 | 735.62 | 735.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 732.70 | 734.65 | 735.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 734.80 | 734.58 | 735.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 734.80 | 734.58 | 735.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 734.80 | 734.58 | 735.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 735.15 | 734.58 | 735.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 736.45 | 734.96 | 735.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 726.85 | 734.96 | 735.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 730.20 | 731.49 | 733.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:30:00 | 733.20 | 732.28 | 733.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 732.20 | 732.28 | 733.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 734.00 | 732.63 | 733.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 738.15 | 732.63 | 733.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 735.05 | 733.11 | 733.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 735.00 | 733.11 | 733.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 735.85 | 733.66 | 733.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 737.85 | 733.66 | 733.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 736.60 | 734.25 | 734.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 736.60 | 734.25 | 734.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 741.80 | 737.19 | 735.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 14:15:00 | 737.45 | 737.69 | 736.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 15:00:00 | 737.45 | 737.69 | 736.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 739.45 | 738.05 | 736.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 750.95 | 738.05 | 736.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 736.00 | 743.26 | 743.21 | SL hit (close<static) qty=1.00 sl=736.50 alert=retest2 |

### Cycle 120 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 732.70 | 741.15 | 742.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 728.15 | 738.55 | 740.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 687.40 | 684.11 | 693.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 687.40 | 684.11 | 693.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 689.00 | 686.22 | 692.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:15:00 | 695.00 | 686.22 | 692.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 695.00 | 687.98 | 693.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 687.25 | 687.98 | 693.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 687.70 | 688.05 | 691.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 14:15:00 | 699.65 | 690.05 | 689.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 699.65 | 690.05 | 689.79 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 681.45 | 691.20 | 691.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 678.50 | 684.83 | 687.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 656.00 | 654.52 | 664.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 668.75 | 654.52 | 664.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 659.75 | 655.56 | 664.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 666.15 | 655.56 | 664.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 655.50 | 654.54 | 659.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 653.65 | 654.43 | 658.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 620.97 | 631.30 | 641.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 632.00 | 630.60 | 639.63 | SL hit (close>ema200) qty=0.50 sl=630.60 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 653.00 | 644.22 | 643.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 663.60 | 648.09 | 645.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 643.85 | 649.76 | 647.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 643.85 | 649.76 | 647.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 643.85 | 649.76 | 647.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 643.50 | 649.76 | 647.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 643.00 | 648.41 | 646.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 643.00 | 648.41 | 646.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 14:15:00 | 644.20 | 645.57 | 645.69 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 662.85 | 649.10 | 647.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 664.40 | 652.16 | 648.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 644.75 | 658.40 | 654.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 644.75 | 658.40 | 654.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 644.75 | 658.40 | 654.42 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 643.65 | 651.30 | 651.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 642.70 | 649.58 | 651.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 622.50 | 621.17 | 631.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 622.50 | 621.17 | 631.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 622.50 | 621.17 | 631.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:30:00 | 618.50 | 621.26 | 627.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 15:15:00 | 633.00 | 630.03 | 629.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 15:15:00 | 633.00 | 630.03 | 629.79 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 619.00 | 627.83 | 628.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 614.40 | 625.14 | 627.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 614.50 | 613.72 | 617.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 614.50 | 613.72 | 617.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 626.50 | 616.44 | 618.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 626.50 | 616.44 | 618.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 624.85 | 618.12 | 618.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 626.80 | 618.12 | 618.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 626.15 | 619.73 | 619.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 627.80 | 623.16 | 621.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 11:15:00 | 625.10 | 625.15 | 622.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:45:00 | 625.75 | 625.15 | 622.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 620.30 | 624.60 | 623.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 620.30 | 624.60 | 623.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 621.30 | 623.94 | 622.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 623.85 | 623.83 | 623.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 620.00 | 622.21 | 622.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 620.00 | 622.21 | 622.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 619.05 | 621.58 | 622.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 621.25 | 621.10 | 621.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 14:15:00 | 621.25 | 621.10 | 621.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 621.25 | 621.10 | 621.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 621.25 | 621.10 | 621.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 622.40 | 621.36 | 621.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 611.85 | 621.36 | 621.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 612.25 | 619.54 | 620.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 605.50 | 614.58 | 617.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 608.00 | 603.35 | 605.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:45:00 | 607.60 | 604.73 | 605.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 577.60 | 583.71 | 587.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 577.22 | 583.71 | 587.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 575.23 | 580.94 | 585.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 590.25 | 582.41 | 584.98 | SL hit (close>ema200) qty=0.50 sl=582.41 alert=retest2 |

### Cycle 131 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 596.70 | 586.80 | 586.63 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 588.70 | 590.41 | 590.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 583.65 | 589.06 | 589.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 550.05 | 544.44 | 553.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 550.05 | 544.44 | 553.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 551.05 | 545.76 | 553.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 558.60 | 545.76 | 553.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 549.50 | 546.51 | 552.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 546.45 | 545.89 | 551.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 547.00 | 546.23 | 551.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:00:00 | 547.00 | 546.38 | 551.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 547.00 | 546.83 | 550.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 537.05 | 531.93 | 535.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 537.80 | 531.93 | 535.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 543.95 | 534.33 | 536.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 550.95 | 534.33 | 536.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 551.55 | 537.77 | 537.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 551.55 | 537.77 | 537.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 557.35 | 543.65 | 540.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 539.20 | 544.82 | 542.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 539.20 | 544.82 | 542.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 539.20 | 544.82 | 542.61 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 537.40 | 545.65 | 545.69 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 14:15:00 | 548.80 | 546.13 | 545.82 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 539.50 | 545.58 | 545.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 528.80 | 539.66 | 542.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 539.30 | 536.97 | 540.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 539.30 | 536.97 | 540.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 539.30 | 536.97 | 540.23 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 548.75 | 540.80 | 540.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 559.80 | 549.10 | 545.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 552.80 | 554.02 | 548.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 13:15:00 | 552.60 | 553.54 | 550.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 552.60 | 553.54 | 550.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:30:00 | 550.35 | 553.54 | 550.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 554.00 | 557.13 | 554.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 554.00 | 557.13 | 554.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 545.00 | 554.70 | 553.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 545.00 | 554.70 | 553.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 547.00 | 553.16 | 552.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 536.60 | 553.16 | 552.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 527.00 | 547.93 | 550.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 524.00 | 543.14 | 548.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 526.65 | 525.60 | 533.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 526.65 | 525.60 | 533.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 533.15 | 527.11 | 533.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 533.15 | 527.11 | 533.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 536.40 | 528.97 | 534.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 536.40 | 528.97 | 534.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 546.75 | 532.52 | 535.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 546.75 | 532.52 | 535.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 541.95 | 534.41 | 535.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 564.30 | 534.41 | 535.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 563.55 | 540.24 | 538.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 567.50 | 545.69 | 541.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 550.30 | 550.93 | 545.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 550.30 | 550.93 | 545.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 539.75 | 548.63 | 545.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 539.75 | 548.63 | 545.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 535.20 | 545.94 | 544.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 535.40 | 545.94 | 544.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 536.45 | 542.30 | 542.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 533.10 | 539.19 | 541.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 527.25 | 522.07 | 528.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 527.25 | 522.07 | 528.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 527.25 | 522.07 | 528.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 523.55 | 522.07 | 528.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 524.90 | 522.63 | 528.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 523.90 | 524.30 | 527.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 517.70 | 520.62 | 522.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 521.85 | 520.55 | 522.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 521.85 | 520.55 | 522.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 525.65 | 521.57 | 522.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:30:00 | 523.00 | 521.57 | 522.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 529.50 | 523.15 | 523.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 529.50 | 523.15 | 523.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 531.75 | 524.87 | 523.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 573.75 | 575.26 | 562.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 582.00 | 575.60 | 564.10 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 558.55 | 571.42 | 567.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 558.55 | 571.42 | 567.68 | SL hit (close<ema400) qty=1.00 sl=567.68 alert=retest1 |

### Cycle 142 — SELL (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 15:15:00 | 565.00 | 566.64 | 566.75 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 575.00 | 568.29 | 567.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 585.65 | 572.64 | 569.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 577.80 | 579.76 | 575.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:15:00 | 572.35 | 579.76 | 575.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 563.00 | 576.41 | 574.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 563.00 | 576.41 | 574.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 564.30 | 573.98 | 573.59 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 11:15:00 | 565.00 | 572.19 | 572.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 12:15:00 | 562.00 | 570.15 | 571.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 573.40 | 566.82 | 569.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 573.40 | 566.82 | 569.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 573.40 | 566.82 | 569.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 573.40 | 566.82 | 569.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 575.65 | 568.58 | 569.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 575.30 | 568.58 | 569.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 575.15 | 571.10 | 570.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 13:15:00 | 582.00 | 573.28 | 571.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 585.95 | 586.55 | 581.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 582.60 | 585.76 | 581.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 582.60 | 585.76 | 581.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 581.35 | 585.76 | 581.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 583.30 | 585.27 | 582.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:45:00 | 582.80 | 585.27 | 582.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 582.80 | 584.77 | 582.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:30:00 | 581.30 | 584.77 | 582.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 581.50 | 584.12 | 582.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 581.50 | 584.12 | 582.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 580.70 | 583.44 | 581.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:45:00 | 578.45 | 583.44 | 581.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 575.00 | 580.55 | 580.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 15:15:00 | 571.25 | 576.31 | 578.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 582.40 | 577.53 | 578.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 582.40 | 577.53 | 578.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 582.40 | 577.53 | 578.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 584.20 | 577.53 | 578.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 579.95 | 578.01 | 578.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:30:00 | 578.60 | 577.78 | 578.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 576.65 | 578.02 | 578.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:00:00 | 575.90 | 576.82 | 577.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 575.00 | 567.05 | 567.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 582.50 | 570.14 | 568.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 582.50 | 570.14 | 568.80 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 569.25 | 571.08 | 571.33 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 576.40 | 572.10 | 571.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 612.25 | 592.70 | 585.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 599.90 | 600.08 | 593.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 599.90 | 600.08 | 593.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-31 10:15:00 | 746.40 | 2024-06-03 09:15:00 | 778.20 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2024-05-31 11:00:00 | 746.05 | 2024-06-03 09:15:00 | 778.20 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2024-05-31 14:15:00 | 745.10 | 2024-06-03 09:15:00 | 778.20 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2024-06-25 14:45:00 | 855.00 | 2024-06-27 09:15:00 | 911.60 | STOP_HIT | 1.00 | -6.62% |
| SELL | retest2 | 2024-06-26 09:30:00 | 856.50 | 2024-06-27 09:15:00 | 911.60 | STOP_HIT | 1.00 | -6.43% |
| SELL | retest2 | 2024-07-10 09:15:00 | 887.50 | 2024-07-15 12:15:00 | 885.40 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-07-18 09:15:00 | 919.60 | 2024-07-19 09:15:00 | 874.90 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2024-07-18 15:15:00 | 892.00 | 2024-07-19 09:15:00 | 874.90 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-07-23 12:15:00 | 844.70 | 2024-07-24 12:15:00 | 886.10 | STOP_HIT | 1.00 | -4.90% |
| SELL | retest2 | 2024-08-13 14:30:00 | 824.45 | 2024-08-20 09:15:00 | 839.40 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-08-19 09:30:00 | 827.10 | 2024-08-20 09:15:00 | 839.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-08-19 11:15:00 | 830.70 | 2024-08-20 09:15:00 | 839.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-09-13 15:15:00 | 996.00 | 2024-09-16 09:15:00 | 977.45 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-09-27 14:45:00 | 1026.95 | 2024-10-03 12:15:00 | 1018.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-09-30 09:15:00 | 1040.00 | 2024-10-03 12:15:00 | 1018.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-10-03 10:15:00 | 1037.90 | 2024-10-03 12:15:00 | 1018.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-10-07 09:15:00 | 1016.30 | 2024-10-09 10:15:00 | 1019.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-10-29 10:45:00 | 906.10 | 2024-10-29 14:15:00 | 924.45 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-10-29 11:15:00 | 904.60 | 2024-10-29 14:15:00 | 924.45 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-10-29 12:00:00 | 904.90 | 2024-10-29 14:15:00 | 924.45 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-10-29 13:15:00 | 903.90 | 2024-10-29 14:15:00 | 924.45 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-11-26 11:15:00 | 1007.10 | 2024-12-03 10:15:00 | 1107.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-26 12:00:00 | 1007.05 | 2024-12-03 10:15:00 | 1107.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-26 12:45:00 | 1008.55 | 2024-12-03 10:15:00 | 1109.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-26 13:45:00 | 1009.10 | 2024-12-03 10:15:00 | 1110.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-28 09:15:00 | 1071.00 | 2024-12-04 13:15:00 | 1178.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-16 15:00:00 | 1183.95 | 2024-12-19 10:15:00 | 1202.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-12-17 10:00:00 | 1184.35 | 2024-12-19 10:15:00 | 1202.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-12-18 14:00:00 | 1181.25 | 2024-12-19 10:15:00 | 1202.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-12-18 15:00:00 | 1183.60 | 2024-12-19 10:15:00 | 1202.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-12-23 14:45:00 | 1220.05 | 2024-12-24 09:15:00 | 1181.00 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-01-02 09:15:00 | 1204.25 | 2025-01-06 14:15:00 | 1185.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-01-06 11:45:00 | 1203.65 | 2025-01-06 14:15:00 | 1185.10 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-01-15 15:00:00 | 949.85 | 2025-01-16 10:15:00 | 980.75 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-01-21 10:15:00 | 952.00 | 2025-01-22 10:15:00 | 904.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 12:00:00 | 954.95 | 2025-01-22 10:15:00 | 907.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 952.00 | 2025-01-23 09:15:00 | 859.46 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2025-01-21 12:00:00 | 954.95 | 2025-01-24 09:15:00 | 856.80 | TARGET_HIT | 0.50 | 10.28% |
| BUY | retest2 | 2025-01-31 09:30:00 | 820.70 | 2025-02-01 11:15:00 | 902.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-07 13:15:00 | 832.05 | 2025-02-10 11:15:00 | 806.95 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2025-02-10 09:15:00 | 833.70 | 2025-02-10 11:15:00 | 806.95 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-02-14 09:15:00 | 752.35 | 2025-02-14 13:15:00 | 714.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 752.35 | 2025-02-14 14:15:00 | 748.00 | STOP_HIT | 0.50 | 0.58% |
| BUY | retest2 | 2025-02-21 11:30:00 | 761.70 | 2025-02-24 13:15:00 | 750.05 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-03-05 12:15:00 | 710.50 | 2025-03-10 12:15:00 | 706.75 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-03-05 12:45:00 | 713.20 | 2025-03-10 12:15:00 | 706.75 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-03-13 10:30:00 | 680.00 | 2025-03-17 09:15:00 | 714.50 | STOP_HIT | 1.00 | -5.07% |
| SELL | retest1 | 2025-04-08 13:45:00 | 664.55 | 2025-04-15 09:15:00 | 699.00 | STOP_HIT | 1.00 | -5.18% |
| SELL | retest1 | 2025-04-08 14:45:00 | 664.90 | 2025-04-15 09:15:00 | 699.00 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest1 | 2025-04-09 09:15:00 | 657.20 | 2025-04-15 09:15:00 | 699.00 | STOP_HIT | 1.00 | -6.36% |
| BUY | retest2 | 2025-04-21 09:15:00 | 719.60 | 2025-04-25 10:15:00 | 715.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-05-02 11:45:00 | 699.90 | 2025-05-07 11:15:00 | 715.75 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-05-02 14:15:00 | 702.65 | 2025-05-07 11:15:00 | 715.75 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-05-02 15:15:00 | 703.00 | 2025-05-07 11:15:00 | 715.75 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-05-06 10:00:00 | 703.10 | 2025-05-07 11:15:00 | 715.75 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-05-22 09:30:00 | 803.45 | 2025-05-27 09:15:00 | 883.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-06 09:30:00 | 876.20 | 2025-06-12 11:15:00 | 888.15 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2025-06-06 10:00:00 | 875.20 | 2025-06-12 11:15:00 | 888.15 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-06-06 11:30:00 | 874.50 | 2025-06-12 11:15:00 | 888.15 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-06-06 12:30:00 | 879.50 | 2025-06-12 11:15:00 | 888.15 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2025-06-16 12:45:00 | 895.65 | 2025-06-16 14:15:00 | 904.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-06-16 14:15:00 | 896.00 | 2025-06-16 14:15:00 | 904.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-06-24 10:45:00 | 888.00 | 2025-06-24 11:15:00 | 894.35 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-25 09:15:00 | 932.65 | 2025-06-27 15:15:00 | 908.80 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-07-07 11:00:00 | 889.30 | 2025-07-09 13:15:00 | 898.55 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-08 09:30:00 | 891.05 | 2025-07-09 13:15:00 | 898.55 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-07-15 11:00:00 | 872.20 | 2025-07-16 11:15:00 | 882.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-16 09:45:00 | 873.40 | 2025-07-16 11:15:00 | 882.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-21 11:15:00 | 876.55 | 2025-07-22 11:15:00 | 879.50 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-07-21 13:00:00 | 876.40 | 2025-07-22 11:15:00 | 879.50 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-21 13:30:00 | 876.30 | 2025-07-22 11:15:00 | 879.50 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-07-22 10:00:00 | 877.00 | 2025-07-22 11:15:00 | 879.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-07-24 11:15:00 | 874.10 | 2025-07-24 12:15:00 | 873.30 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-07-28 12:15:00 | 862.80 | 2025-07-30 12:15:00 | 867.65 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-28 15:00:00 | 862.85 | 2025-07-30 12:15:00 | 867.65 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-29 10:30:00 | 857.05 | 2025-07-30 12:15:00 | 867.65 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-07-29 13:30:00 | 863.70 | 2025-07-30 12:15:00 | 867.65 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-30 09:15:00 | 857.60 | 2025-07-30 12:15:00 | 867.65 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-08-05 13:30:00 | 833.40 | 2025-08-13 11:15:00 | 791.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 13:30:00 | 833.40 | 2025-08-13 13:15:00 | 796.65 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-08-06 10:15:00 | 828.45 | 2025-08-14 09:15:00 | 787.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 10:15:00 | 828.45 | 2025-08-18 09:15:00 | 820.20 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-09-01 11:15:00 | 812.00 | 2025-09-01 13:15:00 | 816.05 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-09-01 12:00:00 | 811.95 | 2025-09-01 13:15:00 | 816.05 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-09-01 12:45:00 | 812.00 | 2025-09-01 13:15:00 | 816.05 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-09 09:15:00 | 869.00 | 2025-09-11 12:15:00 | 858.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-09-11 12:00:00 | 862.85 | 2025-09-11 12:15:00 | 858.70 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-12 11:30:00 | 857.90 | 2025-09-12 12:15:00 | 880.80 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-09-17 09:15:00 | 893.45 | 2025-09-18 09:15:00 | 875.35 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-09-17 10:00:00 | 887.25 | 2025-09-18 09:15:00 | 875.35 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-17 13:45:00 | 887.30 | 2025-09-18 09:15:00 | 875.35 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-22 13:45:00 | 866.75 | 2025-09-23 09:15:00 | 928.70 | STOP_HIT | 1.00 | -7.15% |
| SELL | retest2 | 2025-10-17 10:45:00 | 849.30 | 2025-10-21 13:15:00 | 856.40 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-17 12:15:00 | 848.05 | 2025-10-21 13:15:00 | 856.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-10-20 12:15:00 | 848.85 | 2025-10-21 13:15:00 | 856.40 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-11-25 09:15:00 | 700.30 | 2025-12-02 10:15:00 | 704.95 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-26 09:30:00 | 702.75 | 2025-12-02 10:15:00 | 704.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-11-27 09:45:00 | 705.85 | 2025-12-02 10:15:00 | 704.95 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-12-10 11:15:00 | 683.45 | 2025-12-12 12:15:00 | 691.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-11 10:45:00 | 683.05 | 2025-12-12 12:15:00 | 691.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-12-11 13:45:00 | 684.00 | 2025-12-12 12:15:00 | 691.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-11 15:15:00 | 683.95 | 2025-12-12 12:15:00 | 691.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-18 11:15:00 | 704.55 | 2025-12-18 13:15:00 | 700.25 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-18 11:45:00 | 704.65 | 2025-12-18 13:15:00 | 700.25 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-18 12:30:00 | 705.00 | 2025-12-18 13:15:00 | 700.25 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-30 09:15:00 | 726.85 | 2025-12-31 11:15:00 | 736.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-12-30 12:00:00 | 730.20 | 2025-12-31 11:15:00 | 736.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-30 14:30:00 | 733.20 | 2025-12-31 11:15:00 | 736.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-30 15:00:00 | 732.20 | 2025-12-31 11:15:00 | 736.60 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-01-02 09:15:00 | 750.95 | 2026-01-06 11:15:00 | 736.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-01-13 09:15:00 | 687.25 | 2026-01-14 14:15:00 | 699.65 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-01-13 11:30:00 | 687.70 | 2026-01-14 14:15:00 | 699.65 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-23 10:45:00 | 653.65 | 2026-01-27 13:15:00 | 620.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 653.65 | 2026-01-27 15:15:00 | 632.00 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2026-02-03 14:30:00 | 618.50 | 2026-02-04 15:15:00 | 633.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-02-11 09:15:00 | 623.85 | 2026-02-11 11:15:00 | 620.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-02-13 09:15:00 | 605.50 | 2026-02-24 09:15:00 | 577.60 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2026-02-17 10:30:00 | 608.00 | 2026-02-24 09:15:00 | 577.22 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2026-02-17 12:45:00 | 607.60 | 2026-02-24 12:15:00 | 575.23 | PARTIAL | 0.50 | 5.33% |
| SELL | retest2 | 2026-02-13 09:15:00 | 605.50 | 2026-02-24 14:15:00 | 590.25 | STOP_HIT | 0.50 | 2.52% |
| SELL | retest2 | 2026-02-17 10:30:00 | 608.00 | 2026-02-24 14:15:00 | 590.25 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-02-17 12:45:00 | 607.60 | 2026-02-24 14:15:00 | 590.25 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-03-06 10:30:00 | 546.45 | 2026-03-11 09:15:00 | 551.55 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-03-06 12:15:00 | 547.00 | 2026-03-11 09:15:00 | 551.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-06 13:00:00 | 547.00 | 2026-03-11 09:15:00 | 551.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-06 13:45:00 | 547.00 | 2026-03-11 09:15:00 | 551.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-04-01 10:15:00 | 523.55 | 2026-04-06 12:15:00 | 529.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-01 10:45:00 | 524.90 | 2026-04-06 12:15:00 | 529.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-01 13:30:00 | 523.90 | 2026-04-06 12:15:00 | 529.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-04-06 09:15:00 | 517.70 | 2026-04-06 12:15:00 | 529.50 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2026-04-10 09:30:00 | 582.00 | 2026-04-13 09:15:00 | 558.55 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2026-04-13 10:45:00 | 566.50 | 2026-04-15 15:15:00 | 565.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2026-04-13 14:45:00 | 563.85 | 2026-04-15 15:15:00 | 565.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2026-04-27 11:30:00 | 578.60 | 2026-05-04 11:15:00 | 582.50 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-04-27 13:15:00 | 576.65 | 2026-05-04 11:15:00 | 582.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-04-28 10:00:00 | 575.90 | 2026-05-04 11:15:00 | 582.50 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-05-04 10:45:00 | 575.00 | 2026-05-04 11:15:00 | 582.50 | STOP_HIT | 1.00 | -1.30% |
