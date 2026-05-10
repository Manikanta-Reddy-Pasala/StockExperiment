# Fertilisers and Chemicals Travancore Ltd. (FACT)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 902.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 64 |
| ALERT1 | 44 |
| ALERT2 | 42 |
| ALERT2_SKIP | 23 |
| ALERT3 | 105 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 55 |
| PARTIAL | 22 |
| TARGET_HIT | 9 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 27
- **Target hits / Stop hits / Partials:** 9 / 52 / 22
- **Avg / median % per leg:** 2.78% / 3.80%
- **Sum % (uncompounded):** 230.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 15 | 51.7% | 3 | 23 | 3 | 1.37% | 39.7% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 1 | 4 | 3 | 3.95% | 31.6% |
| BUY @ 3rd Alert (retest2) | 21 | 8 | 38.1% | 2 | 19 | 0 | 0.38% | 8.0% |
| SELL (all) | 54 | 41 | 75.9% | 6 | 29 | 19 | 3.53% | 190.8% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.24% | 8.5% |
| SELL @ 3rd Alert (retest2) | 52 | 39 | 75.0% | 6 | 28 | 18 | 3.51% | 182.3% |
| retest1 (combined) | 10 | 9 | 90.0% | 1 | 5 | 4 | 4.01% | 40.1% |
| retest2 (combined) | 73 | 47 | 64.4% | 8 | 47 | 18 | 2.61% | 190.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 809.90 | 788.25 | 787.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 820.85 | 794.77 | 790.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 820.00 | 821.54 | 812.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 827.65 | 821.54 | 812.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 13:00:00 | 826.60 | 825.49 | 817.56 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 14:15:00 | 827.55 | 825.63 | 818.35 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:15:00 | 869.03 | 840.41 | 827.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:15:00 | 867.93 | 840.41 | 827.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:15:00 | 868.93 | 840.41 | 827.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 874.30 | 874.64 | 865.53 | SL hit (close<ema200) qty=0.50 sl=874.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 874.30 | 874.64 | 865.53 | SL hit (close<ema200) qty=0.50 sl=874.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 874.30 | 874.64 | 865.53 | SL hit (close<ema200) qty=0.50 sl=874.64 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 912.90 | 882.31 | 871.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 871.80 | 882.31 | 871.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 899.35 | 890.61 | 881.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 910.00 | 890.36 | 885.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 902.00 | 892.95 | 886.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 11:30:00 | 900.10 | 895.51 | 889.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 13:15:00 | 900.35 | 896.36 | 890.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 896.10 | 896.83 | 892.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:30:00 | 895.60 | 896.83 | 892.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 895.55 | 896.74 | 893.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 895.00 | 896.74 | 893.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 909.00 | 915.59 | 912.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:30:00 | 916.10 | 915.73 | 912.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:00:00 | 916.30 | 915.73 | 912.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 897.20 | 909.57 | 911.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 897.20 | 909.57 | 911.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 897.20 | 909.57 | 911.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 897.20 | 909.57 | 911.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 897.20 | 909.57 | 911.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 897.20 | 909.57 | 911.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 897.20 | 909.57 | 911.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 882.80 | 898.04 | 904.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 891.20 | 890.76 | 897.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:45:00 | 891.15 | 890.76 | 897.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 896.90 | 891.99 | 897.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:15:00 | 894.05 | 891.99 | 897.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 890.00 | 891.59 | 896.91 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 973.10 | 907.72 | 903.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 984.90 | 923.16 | 910.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 12:15:00 | 1020.65 | 1022.93 | 1001.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 13:00:00 | 1020.65 | 1022.93 | 1001.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1065.00 | 1058.88 | 1051.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 1055.70 | 1058.88 | 1051.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1052.15 | 1057.39 | 1052.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 1052.60 | 1057.39 | 1052.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1051.50 | 1056.22 | 1052.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1051.50 | 1056.22 | 1052.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1026.55 | 1050.28 | 1049.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1026.55 | 1050.28 | 1049.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1035.35 | 1047.30 | 1048.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1018.00 | 1032.23 | 1039.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1028.50 | 1024.41 | 1031.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:30:00 | 1027.20 | 1024.41 | 1031.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1038.85 | 1027.30 | 1032.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1038.85 | 1027.30 | 1032.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1045.20 | 1030.88 | 1033.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 1041.10 | 1030.88 | 1033.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1037.70 | 1033.23 | 1034.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1042.00 | 1033.23 | 1034.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1048.00 | 1036.19 | 1035.34 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 1030.70 | 1035.03 | 1035.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 15:15:00 | 1022.60 | 1032.46 | 1033.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 1037.60 | 1033.48 | 1034.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1037.60 | 1033.48 | 1034.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1037.60 | 1033.48 | 1034.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 1037.60 | 1033.48 | 1034.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1028.65 | 1032.52 | 1033.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 1025.20 | 1031.89 | 1033.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:15:00 | 1023.55 | 1030.85 | 1032.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 1023.05 | 1029.20 | 1031.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 1021.00 | 1027.56 | 1030.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1022.55 | 1025.35 | 1029.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1023.00 | 1025.35 | 1029.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1016.55 | 1007.60 | 1015.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1016.55 | 1007.60 | 1015.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1017.00 | 1009.48 | 1015.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1017.00 | 1009.48 | 1015.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1008.45 | 1009.27 | 1014.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1022.95 | 1017.31 | 1017.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1022.95 | 1017.31 | 1017.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1022.95 | 1017.31 | 1017.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1022.95 | 1017.31 | 1017.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1022.95 | 1017.31 | 1017.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 1029.00 | 1020.51 | 1018.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1026.40 | 1028.54 | 1024.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 1026.40 | 1028.54 | 1024.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1026.40 | 1028.54 | 1024.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 1026.70 | 1028.54 | 1024.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1024.95 | 1027.82 | 1024.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 1018.90 | 1027.82 | 1024.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1018.90 | 1026.04 | 1024.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1034.00 | 1026.04 | 1024.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 991.80 | 1027.20 | 1027.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 991.80 | 1027.20 | 1027.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 975.35 | 1016.83 | 1022.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 974.30 | 973.88 | 986.22 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:30:00 | 968.00 | 973.11 | 984.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 973.00 | 973.59 | 979.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:30:00 | 971.50 | 973.10 | 978.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 970.25 | 972.95 | 978.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 968.50 | 971.42 | 976.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:15:00 | 922.92 | 929.28 | 936.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:15:00 | 921.74 | 929.28 | 936.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 931.05 | 928.86 | 933.89 | SL hit (close>ema200) qty=0.50 sl=928.86 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 931.05 | 928.86 | 933.89 | SL hit (close>ema200) qty=0.50 sl=928.86 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 938.95 | 930.88 | 934.35 | SL hit (close>ema400) qty=1.00 sl=934.35 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:15:00 | 920.07 | 926.30 | 929.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 978.50 | 922.38 | 921.31 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 978.50 | 922.38 | 921.31 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 952.90 | 959.85 | 960.15 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 964.85 | 960.59 | 960.18 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 957.80 | 959.85 | 959.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 954.60 | 958.57 | 959.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 977.00 | 952.42 | 953.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 977.00 | 952.42 | 953.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 977.00 | 952.42 | 953.20 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 973.10 | 956.55 | 955.00 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 948.55 | 958.91 | 960.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 935.00 | 949.35 | 954.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 942.20 | 939.77 | 947.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 942.20 | 939.77 | 947.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 942.20 | 939.77 | 947.10 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 963.40 | 950.27 | 948.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 967.50 | 955.79 | 951.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 951.70 | 958.10 | 954.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 951.70 | 958.10 | 954.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 951.70 | 958.10 | 954.59 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 947.40 | 952.76 | 952.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 943.55 | 950.91 | 951.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 950.30 | 948.60 | 950.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 950.30 | 948.60 | 950.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 950.30 | 948.60 | 950.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 12:15:00 | 939.00 | 947.13 | 949.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 12:45:00 | 936.50 | 945.07 | 948.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 939.65 | 939.26 | 943.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 11:30:00 | 938.80 | 939.29 | 943.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 968.15 | 944.09 | 943.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 968.15 | 944.09 | 943.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 968.15 | 944.09 | 943.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 968.15 | 944.09 | 943.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 968.15 | 944.09 | 943.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 973.85 | 950.04 | 946.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 965.10 | 965.11 | 957.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 965.10 | 965.11 | 957.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 965.10 | 965.11 | 957.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 963.00 | 965.11 | 957.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 942.50 | 960.09 | 956.35 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 941.20 | 954.10 | 954.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 938.60 | 951.00 | 952.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 933.00 | 932.52 | 939.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 933.00 | 932.52 | 939.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 933.00 | 932.52 | 939.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 925.50 | 930.83 | 936.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 944.00 | 930.44 | 928.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 944.00 | 930.44 | 928.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 985.95 | 950.11 | 940.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 963.25 | 967.94 | 956.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 14:30:00 | 966.15 | 967.94 | 956.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 967.00 | 966.59 | 957.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 966.00 | 966.59 | 957.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 962.85 | 964.96 | 959.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 963.40 | 964.96 | 959.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:00:00 | 964.95 | 964.96 | 959.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 982.70 | 1007.66 | 1008.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 982.70 | 1007.66 | 1008.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 982.70 | 1007.66 | 1008.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 965.50 | 983.93 | 994.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 934.70 | 931.69 | 945.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 937.15 | 931.69 | 945.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 947.90 | 936.15 | 943.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 948.00 | 936.15 | 943.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 948.80 | 938.68 | 943.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 950.60 | 938.68 | 943.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 977.00 | 948.79 | 947.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 1011.35 | 961.31 | 953.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 999.00 | 1000.86 | 988.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:30:00 | 997.30 | 1000.86 | 988.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 986.60 | 1000.71 | 997.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 986.60 | 1000.71 | 997.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 990.00 | 998.57 | 996.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 986.90 | 998.57 | 996.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 988.00 | 995.72 | 995.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 988.00 | 995.72 | 995.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 986.20 | 993.82 | 994.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 983.30 | 991.71 | 993.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 981.15 | 974.51 | 980.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 981.15 | 974.51 | 980.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 981.15 | 974.51 | 980.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 980.00 | 974.51 | 980.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 976.70 | 974.95 | 979.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:00:00 | 971.45 | 974.25 | 979.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 972.10 | 973.82 | 978.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 986.85 | 975.79 | 977.75 | SL hit (close>static) qty=1.00 sl=981.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 986.85 | 975.79 | 977.75 | SL hit (close>static) qty=1.00 sl=981.15 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:15:00 | 971.80 | 978.25 | 978.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 981.90 | 977.98 | 978.40 | SL hit (close>static) qty=1.00 sl=981.15 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:45:00 | 973.70 | 977.98 | 978.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 985.90 | 979.57 | 979.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 985.90 | 979.57 | 979.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1001.75 | 984.80 | 982.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 1003.70 | 1004.07 | 998.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 12:00:00 | 1003.70 | 1004.07 | 998.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 992.10 | 1002.17 | 999.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 993.25 | 1002.17 | 999.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 993.20 | 1000.38 | 999.37 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 987.70 | 996.77 | 997.84 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 999.00 | 996.82 | 996.80 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 992.35 | 996.28 | 996.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 990.40 | 994.61 | 995.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 979.80 | 976.01 | 982.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 979.80 | 976.01 | 982.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 979.80 | 976.01 | 982.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 979.80 | 976.01 | 982.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 974.40 | 975.69 | 981.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 972.15 | 975.69 | 981.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 991.10 | 977.85 | 981.57 | SL hit (close>static) qty=1.00 sl=983.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 971.35 | 976.47 | 980.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 970.90 | 976.47 | 980.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 970.35 | 975.82 | 979.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 976.60 | 975.98 | 979.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 975.70 | 975.98 | 979.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:00:00 | 972.60 | 975.30 | 978.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 922.78 | 938.04 | 949.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 922.35 | 938.04 | 949.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 921.83 | 938.04 | 949.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 926.91 | 938.04 | 949.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 923.97 | 938.04 | 949.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-01 10:15:00 | 878.13 | 894.09 | 913.61 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-01 10:15:00 | 875.34 | 894.09 | 913.61 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 892.45 | 888.70 | 901.58 | SL hit (close>ema200) qty=0.50 sl=888.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 892.45 | 888.70 | 901.58 | SL hit (close>ema200) qty=0.50 sl=888.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 892.45 | 888.70 | 901.58 | SL hit (close>ema200) qty=0.50 sl=888.70 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 908.15 | 882.31 | 880.43 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 882.60 | 890.98 | 891.46 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 902.00 | 889.04 | 888.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 914.25 | 894.08 | 890.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 14:15:00 | 900.25 | 900.58 | 895.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 14:30:00 | 899.50 | 900.58 | 895.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 896.65 | 899.79 | 895.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 910.40 | 899.79 | 895.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:15:00 | 900.90 | 903.34 | 901.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 900.95 | 903.07 | 901.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 911.25 | 903.15 | 902.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 901.80 | 905.25 | 904.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 901.80 | 905.25 | 904.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 900.30 | 904.26 | 904.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 900.30 | 904.26 | 904.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 902.15 | 903.83 | 903.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 902.15 | 903.83 | 903.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 902.15 | 903.83 | 903.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 902.15 | 903.83 | 903.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 902.15 | 903.83 | 903.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 897.10 | 902.22 | 903.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 888.45 | 885.80 | 891.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 888.45 | 885.80 | 891.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 888.45 | 885.80 | 891.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 889.10 | 885.80 | 891.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 882.45 | 885.20 | 889.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 888.50 | 885.20 | 889.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 882.80 | 884.81 | 888.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 879.00 | 883.40 | 886.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 892.00 | 881.26 | 882.82 | SL hit (close>static) qty=1.00 sl=890.90 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 896.80 | 886.40 | 885.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 14:15:00 | 903.00 | 893.92 | 890.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 13:15:00 | 903.85 | 904.24 | 898.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 14:00:00 | 903.85 | 904.24 | 898.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 908.10 | 911.36 | 908.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 908.10 | 911.36 | 908.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 909.40 | 910.97 | 908.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 907.60 | 910.97 | 908.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 903.25 | 909.43 | 908.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 903.25 | 909.43 | 908.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 903.05 | 908.15 | 907.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 903.55 | 908.15 | 907.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 902.05 | 906.93 | 907.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 886.80 | 900.48 | 903.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 872.85 | 871.65 | 880.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:30:00 | 871.90 | 871.65 | 880.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 873.75 | 872.64 | 876.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:45:00 | 875.35 | 872.64 | 876.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 875.95 | 873.41 | 875.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 875.95 | 873.41 | 875.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 875.95 | 873.92 | 875.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 884.25 | 873.92 | 875.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 888.00 | 876.73 | 876.92 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 884.65 | 878.32 | 877.62 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 876.10 | 878.24 | 878.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 874.50 | 876.74 | 877.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 13:15:00 | 884.75 | 876.97 | 877.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 13:15:00 | 884.75 | 876.97 | 877.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 884.75 | 876.97 | 877.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:45:00 | 883.70 | 876.97 | 877.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 873.95 | 876.37 | 877.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 871.55 | 876.37 | 877.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 827.97 | 839.29 | 847.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 845.00 | 837.02 | 843.03 | SL hit (close>ema200) qty=0.50 sl=837.02 alert=retest2 |

### Cycle 35 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 820.35 | 815.81 | 815.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 828.10 | 818.27 | 816.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 870.20 | 885.59 | 867.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 09:30:00 | 871.55 | 885.59 | 867.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 867.10 | 877.45 | 867.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:30:00 | 865.15 | 877.45 | 867.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 866.00 | 875.16 | 867.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:30:00 | 866.50 | 875.16 | 867.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 863.00 | 872.73 | 867.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 863.00 | 872.73 | 867.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 861.95 | 870.57 | 866.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 849.00 | 870.57 | 866.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 850.65 | 862.73 | 863.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 10:15:00 | 845.30 | 851.18 | 856.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 853.50 | 851.64 | 855.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 853.50 | 851.64 | 855.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 852.30 | 851.77 | 855.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 854.80 | 851.77 | 855.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 854.85 | 852.39 | 855.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 854.20 | 852.39 | 855.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 860.70 | 854.05 | 856.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 860.70 | 854.05 | 856.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 869.00 | 857.04 | 857.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 866.85 | 857.04 | 857.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 872.05 | 860.04 | 858.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 884.95 | 871.37 | 865.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 878.65 | 884.51 | 878.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 11:15:00 | 878.65 | 884.51 | 878.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 878.65 | 884.51 | 878.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 879.00 | 884.51 | 878.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 877.75 | 882.32 | 878.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 877.25 | 882.32 | 878.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 875.65 | 880.98 | 878.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 875.65 | 880.98 | 878.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 894.75 | 883.02 | 879.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 903.00 | 883.02 | 879.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 914.05 | 887.99 | 884.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 900.60 | 900.03 | 892.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 906.20 | 905.73 | 897.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 912.50 | 913.75 | 908.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 915.30 | 913.80 | 909.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 904.10 | 911.86 | 908.95 | SL hit (close<static) qty=1.00 sl=906.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 904.45 | 907.17 | 907.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 904.45 | 907.17 | 907.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 904.45 | 907.17 | 907.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 904.45 | 907.17 | 907.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 904.45 | 907.17 | 907.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 15:15:00 | 901.00 | 905.94 | 906.87 | Break + close below crossover candle low |

### Cycle 39 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 914.25 | 907.60 | 907.54 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 906.55 | 907.49 | 907.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 897.10 | 905.13 | 906.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 890.75 | 878.80 | 883.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 890.75 | 878.80 | 883.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 890.75 | 878.80 | 883.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 895.65 | 878.80 | 883.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 885.00 | 880.04 | 883.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:45:00 | 892.05 | 880.04 | 883.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 879.60 | 880.11 | 882.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 873.55 | 883.68 | 883.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 877.15 | 881.53 | 882.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 15:15:00 | 833.29 | 843.89 | 853.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 829.87 | 842.49 | 851.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 789.43 | 807.85 | 822.83 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-21 09:15:00 | 786.19 | 805.09 | 820.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 41 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 823.20 | 799.78 | 798.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 825.60 | 816.23 | 812.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 805.85 | 816.28 | 812.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 805.85 | 816.28 | 812.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 805.85 | 816.28 | 812.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 805.85 | 816.28 | 812.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 798.95 | 812.82 | 811.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 801.00 | 812.82 | 811.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 790.80 | 808.41 | 809.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 782.85 | 803.30 | 807.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 777.50 | 772.94 | 786.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 777.50 | 772.94 | 786.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 787.30 | 775.81 | 786.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 792.55 | 775.81 | 786.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 789.95 | 778.64 | 786.84 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 796.70 | 791.47 | 790.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 810.40 | 796.03 | 793.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 795.85 | 798.79 | 796.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 15:15:00 | 795.85 | 798.79 | 796.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 795.85 | 798.79 | 796.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 796.10 | 798.79 | 796.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 788.90 | 796.81 | 795.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 789.15 | 796.81 | 795.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 780.35 | 793.52 | 794.11 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 793.50 | 790.52 | 790.52 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 786.00 | 789.62 | 790.11 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 797.80 | 791.25 | 790.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 811.90 | 795.38 | 792.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 814.15 | 814.46 | 808.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 814.15 | 814.46 | 808.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 802.85 | 812.05 | 808.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 802.85 | 812.05 | 808.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 807.90 | 811.22 | 808.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 811.80 | 808.77 | 807.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 800.05 | 807.51 | 807.40 | SL hit (close<static) qty=1.00 sl=802.85 alert=retest2 |

### Cycle 48 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 800.30 | 806.07 | 806.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 778.20 | 797.74 | 802.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 789.35 | 786.59 | 793.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 789.35 | 786.59 | 793.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 789.35 | 786.59 | 793.02 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 800.30 | 793.57 | 793.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 808.55 | 796.57 | 794.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 802.60 | 803.09 | 798.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 802.60 | 803.09 | 798.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 801.90 | 803.01 | 800.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 801.90 | 803.01 | 800.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 797.25 | 801.70 | 800.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 797.25 | 801.70 | 800.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 796.65 | 800.69 | 799.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 797.00 | 800.69 | 799.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 791.00 | 798.75 | 799.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 790.40 | 797.08 | 798.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 767.25 | 762.40 | 770.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 767.25 | 762.40 | 770.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 767.25 | 762.40 | 770.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:30:00 | 758.60 | 761.71 | 768.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 757.65 | 758.61 | 763.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 720.67 | 738.96 | 745.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 719.77 | 738.96 | 745.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 682.74 | 711.81 | 726.61 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 681.88 | 711.81 | 726.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 744.95 | 687.48 | 682.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 771.00 | 704.18 | 690.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 846.90 | 849.54 | 799.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:15:00 | 912.10 | 849.82 | 821.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 836.75 | 849.08 | 830.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:30:00 | 839.75 | 849.08 | 830.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 816.90 | 843.39 | 834.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 816.90 | 843.39 | 834.04 | SL hit (close<ema400) qty=1.00 sl=834.04 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 805.40 | 843.39 | 834.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 814.30 | 837.57 | 832.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:30:00 | 809.90 | 837.57 | 832.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 808.45 | 827.94 | 828.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 13:15:00 | 804.50 | 823.25 | 826.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 826.20 | 816.57 | 821.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 10:15:00 | 826.20 | 816.57 | 821.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 826.20 | 816.57 | 821.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:30:00 | 802.50 | 813.21 | 817.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 802.50 | 811.07 | 815.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 13:00:00 | 802.20 | 808.05 | 813.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 803.55 | 800.75 | 807.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 803.50 | 801.30 | 807.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 11:30:00 | 798.90 | 801.44 | 806.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 798.00 | 801.44 | 806.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 762.38 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 762.38 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 762.09 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 763.37 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 758.95 | 788.82 | 798.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 758.10 | 782.67 | 794.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 772.00 | 767.16 | 779.84 | SL hit (close>ema200) qty=0.50 sl=767.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 772.00 | 767.16 | 779.84 | SL hit (close>ema200) qty=0.50 sl=767.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 772.00 | 767.16 | 779.84 | SL hit (close>ema200) qty=0.50 sl=767.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 772.00 | 767.16 | 779.84 | SL hit (close>ema200) qty=0.50 sl=767.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 772.00 | 767.16 | 779.84 | SL hit (close>ema200) qty=0.50 sl=767.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 772.00 | 767.16 | 779.84 | SL hit (close>ema200) qty=0.50 sl=767.16 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 794.00 | 783.48 | 783.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 809.35 | 788.65 | 785.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 790.85 | 795.03 | 790.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 790.85 | 795.03 | 790.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 790.85 | 795.03 | 790.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 790.85 | 795.03 | 790.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 794.00 | 794.83 | 791.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 813.35 | 794.83 | 791.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 790.35 | 793.93 | 790.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 784.80 | 793.93 | 790.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 781.45 | 791.43 | 790.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 781.75 | 791.43 | 790.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 782.60 | 789.67 | 789.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:45:00 | 782.00 | 789.67 | 789.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 781.25 | 787.98 | 788.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 763.00 | 782.31 | 785.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 777.90 | 766.00 | 773.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 777.90 | 766.00 | 773.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 777.90 | 766.00 | 773.95 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 786.05 | 778.57 | 778.09 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 755.00 | 775.88 | 777.17 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 782.85 | 776.02 | 775.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 791.95 | 781.47 | 778.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 786.00 | 786.61 | 783.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 804.85 | 786.61 | 783.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 885.34 | 820.92 | 805.43 | Target hit (10%) qty=1.00 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 844.65 | 852.71 | 842.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 846.35 | 852.71 | 842.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 849.45 | 849.80 | 843.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 930.99 | 882.93 | 869.94 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-17 09:15:00 | 934.40 | 882.93 | 869.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 902.95 | 913.89 | 915.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 899.05 | 910.92 | 913.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 904.30 | 901.72 | 907.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 904.30 | 901.72 | 907.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 904.30 | 901.72 | 907.78 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 913.00 | 908.43 | 908.18 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 890.20 | 905.47 | 907.06 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 905.85 | 903.84 | 903.76 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 902.70 | 903.61 | 903.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 11:15:00 | 900.75 | 903.04 | 903.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 905.35 | 903.50 | 903.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 12:15:00 | 905.35 | 903.50 | 903.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 905.35 | 903.50 | 903.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 905.35 | 903.50 | 903.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 900.35 | 902.87 | 903.28 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 910.85 | 904.50 | 903.92 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 905.10 | 908.71 | 908.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 902.80 | 906.80 | 907.93 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:15:00 | 827.65 | 2025-05-15 09:15:00 | 869.03 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 13:00:00 | 826.60 | 2025-05-15 09:15:00 | 867.93 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 14:15:00 | 827.55 | 2025-05-15 09:15:00 | 868.93 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 09:15:00 | 827.65 | 2025-05-19 13:15:00 | 874.30 | STOP_HIT | 0.50 | 5.64% |
| BUY | retest1 | 2025-05-14 13:00:00 | 826.60 | 2025-05-19 13:15:00 | 874.30 | STOP_HIT | 0.50 | 5.77% |
| BUY | retest1 | 2025-05-14 14:15:00 | 827.55 | 2025-05-19 13:15:00 | 874.30 | STOP_HIT | 0.50 | 5.65% |
| BUY | retest2 | 2025-05-22 09:15:00 | 910.00 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-05-22 09:45:00 | 902.00 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-22 11:30:00 | 900.10 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-22 13:15:00 | 900.35 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-05-29 09:30:00 | 916.10 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-05-29 10:00:00 | 916.30 | 2025-05-30 09:15:00 | 897.20 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-18 12:15:00 | 1025.20 | 2025-06-23 10:15:00 | 1022.95 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-06-18 13:15:00 | 1023.55 | 2025-06-23 10:15:00 | 1022.95 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-06-18 13:45:00 | 1023.05 | 2025-06-23 10:15:00 | 1022.95 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-06-18 15:00:00 | 1021.00 | 2025-06-23 10:15:00 | 1022.95 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1034.00 | 2025-06-26 09:15:00 | 991.80 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest1 | 2025-06-30 10:30:00 | 968.00 | 2025-07-08 12:15:00 | 922.92 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2025-07-01 11:30:00 | 971.50 | 2025-07-08 12:15:00 | 921.74 | PARTIAL | 0.50 | 5.12% |
| SELL | retest1 | 2025-06-30 10:30:00 | 968.00 | 2025-07-09 09:15:00 | 931.05 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-07-01 11:30:00 | 971.50 | 2025-07-09 09:15:00 | 931.05 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2025-07-01 13:15:00 | 970.25 | 2025-07-09 10:15:00 | 938.95 | STOP_HIT | 1.00 | 3.23% |
| SELL | retest2 | 2025-07-01 15:00:00 | 968.50 | 2025-07-11 10:15:00 | 920.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 15:00:00 | 968.50 | 2025-07-15 09:15:00 | 978.50 | STOP_HIT | 0.50 | -1.03% |
| SELL | retest2 | 2025-08-01 12:15:00 | 939.00 | 2025-08-05 09:15:00 | 968.15 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-08-01 12:45:00 | 936.50 | 2025-08-05 09:15:00 | 968.15 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-08-04 10:45:00 | 939.65 | 2025-08-05 09:15:00 | 968.15 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-08-04 11:30:00 | 938.80 | 2025-08-05 09:15:00 | 968.15 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-08-08 15:00:00 | 925.50 | 2025-08-12 14:15:00 | 944.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-18 13:30:00 | 963.40 | 2025-08-26 09:15:00 | 982.70 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-08-18 14:00:00 | 964.95 | 2025-08-26 09:15:00 | 982.70 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-09-10 12:00:00 | 971.45 | 2025-09-11 09:15:00 | 986.85 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-10 13:00:00 | 972.10 | 2025-09-11 09:15:00 | 986.85 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-11 13:15:00 | 971.80 | 2025-09-11 14:15:00 | 981.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-11 14:45:00 | 973.70 | 2025-09-11 15:15:00 | 985.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-24 11:15:00 | 972.15 | 2025-09-24 12:15:00 | 991.10 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-24 14:45:00 | 971.35 | 2025-09-29 14:15:00 | 922.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 15:15:00 | 970.90 | 2025-09-29 14:15:00 | 922.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:30:00 | 970.35 | 2025-09-29 14:15:00 | 921.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 11:15:00 | 975.70 | 2025-09-29 14:15:00 | 926.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:00:00 | 972.60 | 2025-09-29 14:15:00 | 923.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:45:00 | 971.35 | 2025-10-01 10:15:00 | 878.13 | TARGET_HIT | 0.50 | 9.60% |
| SELL | retest2 | 2025-09-24 15:15:00 | 970.90 | 2025-10-01 10:15:00 | 875.34 | TARGET_HIT | 0.50 | 9.84% |
| SELL | retest2 | 2025-09-25 09:30:00 | 970.35 | 2025-10-03 09:15:00 | 892.45 | STOP_HIT | 0.50 | 8.03% |
| SELL | retest2 | 2025-09-25 11:15:00 | 975.70 | 2025-10-03 09:15:00 | 892.45 | STOP_HIT | 0.50 | 8.53% |
| SELL | retest2 | 2025-09-25 12:00:00 | 972.60 | 2025-10-03 09:15:00 | 892.45 | STOP_HIT | 0.50 | 8.24% |
| BUY | retest2 | 2025-10-29 09:15:00 | 910.40 | 2025-11-04 11:15:00 | 902.15 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-30 14:15:00 | 900.90 | 2025-11-04 11:15:00 | 902.15 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-10-30 14:45:00 | 900.95 | 2025-11-04 11:15:00 | 902.15 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-11-03 09:15:00 | 911.25 | 2025-11-04 11:15:00 | 902.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-11-11 09:15:00 | 879.00 | 2025-11-12 09:15:00 | 892.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-01 15:15:00 | 871.55 | 2025-12-05 09:15:00 | 827.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 871.55 | 2025-12-05 14:15:00 | 845.00 | STOP_HIT | 0.50 | 3.05% |
| BUY | retest2 | 2025-12-26 10:15:00 | 903.00 | 2026-01-01 09:15:00 | 904.10 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-12-29 09:15:00 | 914.05 | 2026-01-01 14:15:00 | 904.45 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-29 14:15:00 | 900.60 | 2026-01-01 14:15:00 | 904.45 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-12-30 09:30:00 | 906.20 | 2026-01-01 14:15:00 | 904.45 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2026-01-01 09:15:00 | 915.30 | 2026-01-01 14:15:00 | 904.45 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-01-09 09:15:00 | 873.55 | 2026-01-16 15:15:00 | 833.29 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2026-01-09 10:45:00 | 877.15 | 2026-01-19 09:15:00 | 829.87 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2026-01-09 09:15:00 | 873.55 | 2026-01-20 15:15:00 | 789.43 | TARGET_HIT | 0.50 | 9.63% |
| SELL | retest2 | 2026-01-09 10:45:00 | 877.15 | 2026-01-21 09:15:00 | 786.19 | TARGET_HIT | 0.50 | 10.37% |
| BUY | retest2 | 2026-02-11 15:15:00 | 811.80 | 2026-02-12 09:15:00 | 800.05 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-25 11:30:00 | 758.60 | 2026-03-02 09:15:00 | 720.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:45:00 | 757.65 | 2026-03-02 09:15:00 | 719.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:30:00 | 758.60 | 2026-03-04 09:15:00 | 682.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 10:45:00 | 757.65 | 2026-03-04 09:15:00 | 681.88 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-03-13 09:15:00 | 912.10 | 2026-03-16 09:15:00 | 816.90 | STOP_HIT | 1.00 | -10.44% |
| SELL | retest2 | 2026-03-19 09:30:00 | 802.50 | 2026-03-23 09:15:00 | 762.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 11:00:00 | 802.50 | 2026-03-23 09:15:00 | 762.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 13:00:00 | 802.20 | 2026-03-23 09:15:00 | 762.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 09:30:00 | 803.55 | 2026-03-23 09:15:00 | 763.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 11:30:00 | 798.90 | 2026-03-23 09:15:00 | 758.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 798.00 | 2026-03-23 10:15:00 | 758.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:30:00 | 802.50 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2026-03-19 11:00:00 | 802.50 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2026-03-19 13:00:00 | 802.20 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2026-03-20 09:30:00 | 803.55 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2026-03-20 11:30:00 | 798.90 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2026-03-20 12:15:00 | 798.00 | 2026-03-24 09:15:00 | 772.00 | STOP_HIT | 0.50 | 3.26% |
| BUY | retest1 | 2026-04-08 09:15:00 | 804.85 | 2026-04-09 09:15:00 | 885.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 846.35 | 2026-04-17 09:15:00 | 930.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:45:00 | 849.45 | 2026-04-17 09:15:00 | 934.40 | TARGET_HIT | 1.00 | 10.00% |
