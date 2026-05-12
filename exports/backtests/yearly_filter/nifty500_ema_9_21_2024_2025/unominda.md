# UNO Minda Ltd. (UNOMINDA)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 1179.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 144 |
| ALERT1 | 106 |
| ALERT2 | 105 |
| ALERT2_SKIP | 53 |
| ALERT3 | 286 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 143 |
| PARTIAL | 10 |
| TARGET_HIT | 12 |
| STOP_HIT | 132 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 154 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 110
- **Target hits / Stop hits / Partials:** 12 / 132 / 10
- **Avg / median % per leg:** 0.29% / -1.21%
- **Sum % (uncompounded):** 45.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 67 | 18 | 26.9% | 11 | 56 | 0 | 0.83% | 55.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 67 | 18 | 26.9% | 11 | 56 | 0 | 0.83% | 55.5% |
| SELL (all) | 87 | 26 | 29.9% | 1 | 76 | 10 | -0.12% | -10.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.27% | -0.3% |
| SELL @ 3rd Alert (retest2) | 86 | 26 | 30.2% | 1 | 75 | 10 | -0.12% | -10.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.27% | -0.3% |
| retest2 (combined) | 153 | 44 | 28.8% | 12 | 131 | 10 | 0.30% | 45.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 15:15:00 | 729.05 | 732.80 | 733.23 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 740.85 | 734.41 | 733.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 752.65 | 738.06 | 735.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 760.00 | 760.91 | 752.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 14:45:00 | 760.55 | 760.91 | 752.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 751.70 | 758.08 | 753.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:00:00 | 751.70 | 758.08 | 753.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 753.10 | 757.09 | 753.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:30:00 | 753.00 | 757.09 | 753.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 753.60 | 756.39 | 753.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:45:00 | 754.10 | 756.39 | 753.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 755.80 | 756.27 | 754.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:15:00 | 760.00 | 756.27 | 754.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 10:30:00 | 760.95 | 759.02 | 755.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 15:15:00 | 836.00 | 785.78 | 769.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 13:15:00 | 846.65 | 849.80 | 849.94 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 859.15 | 851.57 | 850.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 883.00 | 859.83 | 855.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 830.15 | 856.61 | 855.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 830.15 | 856.61 | 855.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 830.15 | 856.61 | 855.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 830.15 | 856.61 | 855.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 828.90 | 851.07 | 852.66 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 866.00 | 853.30 | 852.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 15:15:00 | 869.50 | 860.12 | 856.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 962.30 | 966.03 | 941.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 15:00:00 | 962.30 | 966.03 | 941.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 959.70 | 971.41 | 956.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:00:00 | 959.70 | 971.41 | 956.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 945.90 | 966.31 | 955.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 15:00:00 | 945.90 | 966.31 | 955.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 954.00 | 963.85 | 954.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:30:00 | 970.75 | 962.52 | 955.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:15:00 | 960.50 | 962.52 | 955.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 1056.55 | 1033.82 | 1016.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 1063.35 | 1073.96 | 1075.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 1057.30 | 1070.63 | 1073.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 12:15:00 | 1063.60 | 1059.81 | 1065.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 1063.60 | 1059.81 | 1065.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1063.60 | 1059.81 | 1065.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:45:00 | 1072.15 | 1059.81 | 1065.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 1064.40 | 1060.73 | 1065.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:30:00 | 1072.05 | 1060.73 | 1065.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 1055.05 | 1059.59 | 1064.49 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 1084.15 | 1066.81 | 1066.65 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 1061.65 | 1065.82 | 1066.30 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 1073.70 | 1067.39 | 1066.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 15:15:00 | 1079.00 | 1069.71 | 1068.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 1148.55 | 1153.45 | 1132.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 15:00:00 | 1148.55 | 1153.45 | 1132.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1158.60 | 1161.71 | 1154.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:45:00 | 1151.50 | 1161.71 | 1154.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1158.00 | 1161.33 | 1155.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:00:00 | 1158.00 | 1161.33 | 1155.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1158.75 | 1160.82 | 1156.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:30:00 | 1153.25 | 1160.82 | 1156.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1153.85 | 1159.42 | 1155.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:00:00 | 1153.85 | 1159.42 | 1155.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1154.95 | 1158.53 | 1155.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:45:00 | 1153.85 | 1158.53 | 1155.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 1151.15 | 1157.05 | 1155.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:45:00 | 1149.35 | 1157.05 | 1155.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1142.60 | 1157.72 | 1156.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1142.60 | 1157.72 | 1156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 1135.15 | 1153.21 | 1154.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 12:15:00 | 1133.00 | 1149.17 | 1152.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 13:15:00 | 1028.50 | 1027.47 | 1042.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 13:45:00 | 1027.60 | 1027.47 | 1042.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1038.30 | 1028.71 | 1039.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 1035.00 | 1028.71 | 1039.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1040.40 | 1031.05 | 1039.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:45:00 | 1031.50 | 1032.93 | 1038.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 979.92 | 997.51 | 1013.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 974.25 | 967.62 | 986.86 | SL hit (close>ema200) qty=0.50 sl=967.62 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 1001.65 | 987.03 | 986.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 13:15:00 | 1017.85 | 993.19 | 989.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 13:15:00 | 1019.80 | 1020.20 | 1012.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 13:45:00 | 1022.75 | 1020.20 | 1012.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1012.90 | 1018.02 | 1012.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 1010.80 | 1018.02 | 1012.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1012.30 | 1016.88 | 1012.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:15:00 | 1025.25 | 1017.57 | 1013.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 09:15:00 | 1038.00 | 1049.82 | 1049.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 1038.00 | 1049.82 | 1049.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 1021.30 | 1042.44 | 1046.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 14:15:00 | 1037.90 | 1036.79 | 1042.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-01 15:00:00 | 1037.90 | 1036.79 | 1042.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1038.00 | 1037.03 | 1041.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 1027.00 | 1037.03 | 1041.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-05 09:15:00 | 924.30 | 1015.15 | 1026.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 13:15:00 | 1000.60 | 987.58 | 985.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 14:15:00 | 1006.05 | 991.27 | 987.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 1159.35 | 1162.11 | 1137.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:45:00 | 1158.05 | 1162.11 | 1137.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1174.00 | 1162.18 | 1147.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:30:00 | 1186.15 | 1165.76 | 1150.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 12:15:00 | 1139.40 | 1157.46 | 1155.61 | SL hit (close<static) qty=1.00 sl=1146.70 alert=retest2 |

### Cycle 15 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 1140.05 | 1152.42 | 1153.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 15:15:00 | 1130.50 | 1148.04 | 1151.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 13:15:00 | 1096.70 | 1095.96 | 1107.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 14:00:00 | 1096.70 | 1095.96 | 1107.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 1116.00 | 1099.97 | 1108.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 15:00:00 | 1116.00 | 1099.97 | 1108.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1115.10 | 1103.00 | 1109.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 1110.40 | 1103.00 | 1109.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 1104.50 | 1104.09 | 1108.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:30:00 | 1105.00 | 1104.09 | 1108.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 1103.15 | 1103.90 | 1108.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 1098.90 | 1103.16 | 1107.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 13:15:00 | 1120.70 | 1106.67 | 1108.60 | SL hit (close>static) qty=1.00 sl=1109.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 1123.35 | 1110.01 | 1109.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 09:15:00 | 1156.00 | 1123.12 | 1116.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 12:15:00 | 1125.95 | 1129.18 | 1121.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 12:45:00 | 1128.20 | 1129.18 | 1121.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 1125.75 | 1128.50 | 1121.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:00:00 | 1125.75 | 1128.50 | 1121.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1126.35 | 1128.07 | 1122.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 09:15:00 | 1129.20 | 1127.79 | 1122.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 10:00:00 | 1130.05 | 1128.24 | 1123.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-02 09:15:00 | 1242.12 | 1173.17 | 1151.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 1141.25 | 1163.94 | 1165.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1113.00 | 1137.86 | 1150.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 14:15:00 | 1029.00 | 1029.00 | 1049.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-11 15:00:00 | 1029.00 | 1029.00 | 1049.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 1029.00 | 1020.99 | 1032.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 1063.75 | 1020.99 | 1032.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1046.40 | 1026.07 | 1034.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:30:00 | 1052.50 | 1026.07 | 1034.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 1040.80 | 1029.02 | 1034.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:15:00 | 1050.90 | 1029.02 | 1034.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 1045.00 | 1032.21 | 1035.65 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 14:15:00 | 1056.50 | 1041.04 | 1039.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 1066.70 | 1048.73 | 1043.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 1060.80 | 1062.71 | 1053.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 1060.80 | 1062.71 | 1053.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1049.95 | 1059.92 | 1053.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 1049.95 | 1059.92 | 1053.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1039.20 | 1055.77 | 1052.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:45:00 | 1039.00 | 1055.77 | 1052.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 1052.00 | 1051.91 | 1051.29 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 1049.55 | 1050.83 | 1050.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 1032.75 | 1045.49 | 1048.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1040.80 | 1039.16 | 1043.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1040.80 | 1039.16 | 1043.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1040.80 | 1039.16 | 1043.83 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 1060.95 | 1047.19 | 1045.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 1085.95 | 1054.94 | 1049.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 15:15:00 | 1145.55 | 1146.92 | 1128.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 09:15:00 | 1151.65 | 1146.92 | 1128.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1149.20 | 1149.29 | 1141.12 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 11:15:00 | 1115.80 | 1136.97 | 1139.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 11:15:00 | 1100.20 | 1115.79 | 1125.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 992.95 | 992.09 | 1012.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 14:30:00 | 994.00 | 992.09 | 1012.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1011.25 | 996.09 | 1010.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 1011.25 | 996.09 | 1010.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1025.10 | 1001.89 | 1011.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 1022.35 | 1001.89 | 1011.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 1007.65 | 1003.04 | 1011.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 10:15:00 | 1001.05 | 1007.38 | 1010.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:30:00 | 1001.00 | 1004.84 | 1008.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 15:00:00 | 1001.10 | 1003.33 | 1007.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 1016.55 | 1003.04 | 1001.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 1016.55 | 1003.04 | 1001.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 15:15:00 | 1023.15 | 1011.60 | 1006.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 1010.05 | 1011.29 | 1006.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 10:00:00 | 1010.05 | 1011.29 | 1006.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1001.15 | 1009.27 | 1006.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 1001.15 | 1009.27 | 1006.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 993.25 | 1006.06 | 1005.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 993.25 | 1006.06 | 1005.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 984.85 | 1001.82 | 1003.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 980.00 | 992.30 | 996.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 960.70 | 960.00 | 972.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 14:00:00 | 960.70 | 960.00 | 972.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 951.85 | 956.97 | 967.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 944.45 | 954.34 | 964.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:30:00 | 944.50 | 951.18 | 961.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 944.20 | 951.18 | 961.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 13:45:00 | 944.75 | 932.99 | 938.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 935.65 | 933.52 | 938.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:45:00 | 939.90 | 933.52 | 938.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 930.00 | 932.82 | 937.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 939.50 | 932.82 | 937.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 939.75 | 934.20 | 937.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-24 12:15:00 | 948.20 | 939.52 | 939.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 12:15:00 | 948.20 | 939.52 | 939.52 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 921.25 | 939.31 | 939.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 915.50 | 927.91 | 932.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 12:15:00 | 933.85 | 927.74 | 931.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 12:15:00 | 933.85 | 927.74 | 931.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 933.85 | 927.74 | 931.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:30:00 | 933.55 | 927.74 | 931.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 932.75 | 928.74 | 931.54 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 10:15:00 | 939.60 | 933.72 | 933.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 956.85 | 938.35 | 935.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 946.95 | 955.51 | 949.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 13:15:00 | 946.95 | 955.51 | 949.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 946.95 | 955.51 | 949.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 946.95 | 955.51 | 949.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 964.10 | 957.23 | 950.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 15:15:00 | 969.50 | 957.23 | 950.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 09:30:00 | 967.80 | 964.69 | 955.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:30:00 | 966.90 | 973.41 | 971.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 955.05 | 967.44 | 968.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 955.05 | 967.44 | 968.57 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 974.90 | 963.90 | 963.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 978.95 | 966.91 | 964.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 12:15:00 | 960.80 | 972.23 | 969.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 12:15:00 | 960.80 | 972.23 | 969.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 960.80 | 972.23 | 969.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 960.80 | 972.23 | 969.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 957.75 | 969.33 | 968.30 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 949.35 | 965.34 | 966.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 943.10 | 957.56 | 962.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 937.65 | 932.80 | 940.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 937.65 | 932.80 | 940.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 937.65 | 932.80 | 940.90 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 988.30 | 946.61 | 945.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 14:15:00 | 1005.40 | 968.92 | 957.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 951.45 | 967.91 | 958.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 951.45 | 967.91 | 958.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 951.45 | 967.91 | 958.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 951.45 | 967.91 | 958.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 957.00 | 965.72 | 958.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 12:00:00 | 968.35 | 966.25 | 959.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 15:15:00 | 965.35 | 965.71 | 961.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 10:15:00 | 973.70 | 965.24 | 961.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-21 09:15:00 | 1061.89 | 1034.56 | 1014.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 12:15:00 | 1068.15 | 1073.48 | 1073.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 13:15:00 | 1058.25 | 1070.43 | 1072.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 1048.90 | 1032.75 | 1042.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 14:15:00 | 1048.90 | 1032.75 | 1042.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 1048.90 | 1032.75 | 1042.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 1048.90 | 1032.75 | 1042.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 1055.00 | 1037.20 | 1043.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 1038.60 | 1037.20 | 1043.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 15:15:00 | 1060.50 | 1036.76 | 1036.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 1060.50 | 1036.76 | 1036.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 1084.00 | 1046.21 | 1040.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 1086.10 | 1088.58 | 1076.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 10:00:00 | 1086.10 | 1088.58 | 1076.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1069.05 | 1084.83 | 1081.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:45:00 | 1074.40 | 1084.83 | 1081.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1074.25 | 1082.72 | 1080.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:15:00 | 1078.35 | 1080.28 | 1079.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 12:15:00 | 1074.20 | 1079.07 | 1079.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 12:15:00 | 1074.20 | 1079.07 | 1079.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 15:15:00 | 1070.00 | 1076.27 | 1077.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 15:15:00 | 1085.30 | 1070.56 | 1072.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 15:15:00 | 1085.30 | 1070.56 | 1072.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1085.30 | 1070.56 | 1072.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 1083.25 | 1072.83 | 1073.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1077.05 | 1073.67 | 1073.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 11:15:00 | 1071.10 | 1073.67 | 1073.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 11:15:00 | 1077.40 | 1074.42 | 1074.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 1077.40 | 1074.42 | 1074.28 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 1059.50 | 1071.85 | 1073.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 1058.55 | 1069.19 | 1071.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 15:15:00 | 1065.00 | 1064.42 | 1067.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:15:00 | 1054.90 | 1064.42 | 1067.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1053.10 | 1047.38 | 1054.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-16 12:15:00 | 1057.70 | 1050.80 | 1054.60 | SL hit (close>ema400) qty=1.00 sl=1054.60 alert=retest1 |

### Cycle 36 — BUY (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 13:15:00 | 1044.65 | 1039.12 | 1038.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 15:15:00 | 1049.15 | 1042.32 | 1040.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 1049.40 | 1060.01 | 1052.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 1049.40 | 1060.01 | 1052.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1049.40 | 1060.01 | 1052.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1049.40 | 1060.01 | 1052.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1057.00 | 1059.41 | 1052.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 1030.95 | 1059.41 | 1052.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1039.20 | 1055.37 | 1051.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 1026.90 | 1055.37 | 1051.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 11:15:00 | 1027.05 | 1045.07 | 1047.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 14:15:00 | 1021.60 | 1037.68 | 1043.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 12:15:00 | 1047.60 | 1033.71 | 1038.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 12:15:00 | 1047.60 | 1033.71 | 1038.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 1047.60 | 1033.71 | 1038.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:00:00 | 1047.60 | 1033.71 | 1038.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 1042.55 | 1035.48 | 1038.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:30:00 | 1047.80 | 1035.48 | 1038.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 1064.45 | 1041.27 | 1041.08 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 10:15:00 | 1033.50 | 1040.39 | 1040.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 1028.05 | 1037.92 | 1039.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 1039.65 | 1036.39 | 1038.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 14:15:00 | 1039.65 | 1036.39 | 1038.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1039.65 | 1036.39 | 1038.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1039.65 | 1036.39 | 1038.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1052.05 | 1039.52 | 1039.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 09:15:00 | 1031.70 | 1039.52 | 1039.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:15:00 | 1033.00 | 1032.53 | 1034.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 1034.65 | 1033.50 | 1034.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 1029.45 | 1034.40 | 1034.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1028.30 | 1033.18 | 1033.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 1050.00 | 1036.93 | 1035.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 1050.00 | 1036.93 | 1035.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 15:15:00 | 1059.00 | 1044.02 | 1039.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 10:15:00 | 1044.85 | 1045.44 | 1040.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 11:00:00 | 1044.85 | 1045.44 | 1040.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 1044.15 | 1045.18 | 1040.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:45:00 | 1040.75 | 1045.18 | 1040.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 1063.05 | 1048.56 | 1043.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 1063.05 | 1048.56 | 1043.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1061.60 | 1055.97 | 1051.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 14:00:00 | 1063.35 | 1058.34 | 1053.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 14:30:00 | 1063.30 | 1060.16 | 1055.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 1071.90 | 1062.33 | 1057.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:30:00 | 1066.50 | 1064.03 | 1059.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 1062.95 | 1063.74 | 1059.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 1064.95 | 1063.74 | 1059.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 1063.00 | 1063.59 | 1060.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 1044.70 | 1063.59 | 1060.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1057.80 | 1062.44 | 1059.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-07 12:15:00 | 1052.80 | 1058.35 | 1058.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 12:15:00 | 1052.80 | 1058.35 | 1058.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 13:15:00 | 1042.10 | 1055.10 | 1057.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 14:15:00 | 1059.70 | 1056.02 | 1057.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 14:15:00 | 1059.70 | 1056.02 | 1057.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1059.70 | 1056.02 | 1057.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:30:00 | 1065.60 | 1056.02 | 1057.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1059.00 | 1056.62 | 1057.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 1056.10 | 1056.62 | 1057.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 09:15:00 | 1074.55 | 1060.20 | 1058.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 10:15:00 | 1079.75 | 1064.11 | 1060.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 1069.00 | 1070.65 | 1067.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 14:15:00 | 1069.00 | 1070.65 | 1067.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 1069.00 | 1070.65 | 1067.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:30:00 | 1067.65 | 1070.65 | 1067.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 1060.00 | 1068.52 | 1067.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 1059.80 | 1068.52 | 1067.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1059.20 | 1066.66 | 1066.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 1059.20 | 1066.66 | 1066.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1076.85 | 1068.69 | 1067.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 1080.00 | 1068.69 | 1067.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:00:00 | 1078.15 | 1070.93 | 1068.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:30:00 | 1079.95 | 1070.23 | 1068.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 14:45:00 | 1078.90 | 1074.56 | 1070.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1085.50 | 1080.98 | 1074.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 1070.00 | 1080.98 | 1074.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 1073.20 | 1079.42 | 1074.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:00:00 | 1073.20 | 1079.42 | 1074.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 1053.50 | 1074.24 | 1072.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-13 11:15:00 | 1053.50 | 1074.24 | 1072.45 | SL hit (close<static) qty=1.00 sl=1057.05 alert=retest2 |

### Cycle 43 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 1057.85 | 1070.96 | 1071.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 14:15:00 | 1031.10 | 1059.62 | 1065.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1065.00 | 1056.68 | 1063.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1065.00 | 1056.68 | 1063.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1065.00 | 1056.68 | 1063.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 1065.00 | 1056.68 | 1063.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1069.10 | 1059.16 | 1063.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1069.10 | 1059.16 | 1063.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1069.75 | 1061.28 | 1064.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 1069.75 | 1061.28 | 1064.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 1059.70 | 1060.96 | 1063.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:30:00 | 1055.00 | 1056.97 | 1061.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 10:15:00 | 1050.00 | 1055.82 | 1059.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:30:00 | 1057.65 | 1057.16 | 1059.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 1088.70 | 1062.05 | 1060.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1088.70 | 1062.05 | 1060.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 1118.50 | 1073.34 | 1066.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 1088.25 | 1088.28 | 1078.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 1088.25 | 1088.28 | 1078.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1088.25 | 1088.28 | 1078.12 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 1031.50 | 1072.91 | 1075.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1020.20 | 1044.08 | 1057.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 885.70 | 884.76 | 910.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 885.70 | 884.76 | 910.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 902.05 | 891.18 | 904.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 908.85 | 891.18 | 904.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 897.10 | 892.37 | 903.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 896.50 | 892.37 | 903.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:30:00 | 894.95 | 893.16 | 902.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 911.80 | 897.33 | 901.23 | SL hit (close>static) qty=1.00 sl=910.45 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 912.00 | 904.51 | 903.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 934.30 | 910.47 | 906.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 10:15:00 | 958.50 | 964.53 | 948.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 10:30:00 | 955.65 | 964.53 | 948.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 958.30 | 961.19 | 949.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:15:00 | 960.15 | 961.19 | 949.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:45:00 | 960.20 | 961.02 | 950.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-07 09:15:00 | 1056.16 | 1029.31 | 1013.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 12:15:00 | 994.65 | 1029.62 | 1034.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 13:15:00 | 992.40 | 1022.18 | 1030.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1011.40 | 1003.18 | 1017.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1011.40 | 1003.18 | 1017.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1011.40 | 1003.18 | 1017.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 1015.95 | 1003.18 | 1017.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1003.15 | 1003.18 | 1015.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1010.50 | 1003.18 | 1015.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 867.70 | 862.96 | 872.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 867.70 | 862.96 | 872.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 875.85 | 865.54 | 872.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:00:00 | 875.85 | 865.54 | 872.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 867.35 | 865.90 | 871.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:45:00 | 879.60 | 865.90 | 871.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 866.90 | 866.10 | 871.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 875.25 | 866.10 | 871.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 846.30 | 862.14 | 869.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 11:15:00 | 844.45 | 859.22 | 867.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 12:00:00 | 843.00 | 855.98 | 865.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 13:15:00 | 844.70 | 854.14 | 863.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 844.60 | 852.67 | 861.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 857.60 | 850.56 | 857.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 854.05 | 850.56 | 857.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 858.45 | 852.14 | 857.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:30:00 | 858.60 | 852.14 | 857.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 854.35 | 852.58 | 857.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 14:30:00 | 849.65 | 852.96 | 856.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 15:15:00 | 847.70 | 852.96 | 856.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 10:45:00 | 850.10 | 850.11 | 854.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 834.00 | 850.36 | 852.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 826.75 | 845.64 | 850.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 10:45:00 | 823.25 | 841.74 | 848.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:00:00 | 820.15 | 834.65 | 843.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 821.90 | 832.47 | 840.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 10:45:00 | 821.30 | 825.92 | 835.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 11:15:00 | 807.17 | 822.99 | 833.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 11:15:00 | 807.60 | 822.99 | 833.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 12:15:00 | 825.00 | 823.40 | 832.95 | SL hit (close>ema200) qty=0.50 sl=823.40 alert=retest2 |

### Cycle 48 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 832.50 | 827.05 | 826.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 15:15:00 | 842.70 | 831.02 | 828.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 870.55 | 872.22 | 862.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 870.55 | 872.22 | 862.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 863.20 | 870.42 | 862.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 860.85 | 870.42 | 862.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 862.20 | 868.78 | 862.82 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 857.00 | 860.03 | 860.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 850.85 | 858.19 | 859.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 855.40 | 854.45 | 856.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 855.40 | 854.45 | 856.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 855.40 | 854.45 | 856.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 855.40 | 854.45 | 856.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 858.60 | 855.28 | 856.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 854.30 | 855.28 | 856.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 855.10 | 855.25 | 856.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 846.65 | 855.25 | 856.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 850.80 | 855.11 | 856.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:15:00 | 852.80 | 854.91 | 856.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:45:00 | 852.60 | 854.82 | 856.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 862.70 | 856.39 | 856.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 862.70 | 856.39 | 856.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 852.00 | 855.52 | 856.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 861.40 | 855.52 | 856.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 873.35 | 859.08 | 857.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 873.35 | 859.08 | 857.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 14:15:00 | 910.15 | 879.29 | 868.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 12:15:00 | 1002.80 | 1002.91 | 980.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 12:45:00 | 1002.30 | 1002.91 | 980.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 979.25 | 996.88 | 981.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 15:00:00 | 979.25 | 996.88 | 981.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 966.95 | 990.89 | 980.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:15:00 | 971.55 | 990.89 | 980.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 12:15:00 | 962.30 | 975.05 | 975.47 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 15:15:00 | 983.00 | 975.50 | 975.41 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 12:15:00 | 951.15 | 972.23 | 974.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 14:15:00 | 950.05 | 965.14 | 970.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 920.00 | 919.03 | 930.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 09:15:00 | 918.80 | 919.03 | 930.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 918.10 | 918.84 | 929.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:15:00 | 912.55 | 918.84 | 929.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 14:15:00 | 866.92 | 901.69 | 916.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-01 13:15:00 | 897.05 | 891.12 | 903.20 | SL hit (close>ema200) qty=0.50 sl=891.12 alert=retest2 |

### Cycle 54 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 869.90 | 832.25 | 827.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 883.35 | 842.47 | 832.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 864.00 | 866.25 | 852.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 864.00 | 866.25 | 852.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 850.00 | 859.68 | 854.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:15:00 | 857.05 | 858.18 | 854.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 880.80 | 894.76 | 895.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 880.80 | 894.76 | 895.21 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 892.05 | 890.19 | 890.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 14:15:00 | 895.95 | 891.34 | 890.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 884.20 | 890.46 | 890.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 884.20 | 890.46 | 890.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 884.20 | 890.46 | 890.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 11:00:00 | 895.50 | 891.47 | 890.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 12:15:00 | 900.80 | 891.93 | 891.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:30:00 | 898.15 | 892.92 | 892.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 10:15:00 | 886.15 | 891.56 | 891.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 886.15 | 891.56 | 891.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 878.10 | 888.87 | 890.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 885.15 | 881.91 | 885.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 885.15 | 881.91 | 885.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 885.15 | 881.91 | 885.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:15:00 | 891.30 | 881.91 | 885.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 891.30 | 883.79 | 886.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 889.60 | 883.79 | 886.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 893.70 | 885.77 | 886.87 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 898.95 | 889.77 | 888.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 911.75 | 894.17 | 890.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 907.30 | 913.39 | 904.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 15:00:00 | 907.30 | 913.39 | 904.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 907.10 | 912.13 | 904.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 908.55 | 912.13 | 904.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 910.20 | 911.74 | 905.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:45:00 | 916.60 | 913.41 | 907.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 13:30:00 | 917.45 | 915.02 | 909.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:30:00 | 917.70 | 920.71 | 920.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 12:15:00 | 911.30 | 918.83 | 919.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 12:15:00 | 911.30 | 918.83 | 919.55 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 940.00 | 920.70 | 919.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 950.75 | 930.15 | 924.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 948.30 | 948.43 | 939.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 948.30 | 948.43 | 939.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 990.00 | 981.76 | 967.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:15:00 | 994.60 | 981.76 | 967.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 1001.35 | 991.86 | 979.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 994.60 | 991.99 | 981.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:45:00 | 994.45 | 991.05 | 983.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 993.00 | 989.88 | 985.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 982.50 | 989.88 | 985.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1000.00 | 1002.90 | 995.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 996.30 | 1002.90 | 995.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 993.05 | 1000.06 | 995.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 993.05 | 1000.06 | 995.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 988.35 | 997.71 | 994.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 984.85 | 997.71 | 994.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 991.00 | 996.37 | 994.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 980.05 | 993.11 | 993.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 980.05 | 993.11 | 993.26 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 1009.85 | 994.65 | 993.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 1013.60 | 1005.13 | 999.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 15:15:00 | 980.00 | 1001.18 | 998.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 15:15:00 | 980.00 | 1001.18 | 998.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 980.00 | 1001.18 | 998.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 976.10 | 1001.18 | 998.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1005.60 | 1002.06 | 999.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 984.95 | 1002.06 | 999.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1015.25 | 1015.55 | 1009.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:15:00 | 1018.80 | 1015.78 | 1010.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:45:00 | 1018.45 | 1016.23 | 1011.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 1006.50 | 1014.29 | 1010.65 | SL hit (close<static) qty=1.00 sl=1008.10 alert=retest2 |

### Cycle 63 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 999.00 | 1009.58 | 1010.58 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 1016.35 | 1010.33 | 1010.31 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 1000.50 | 1008.36 | 1009.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 999.00 | 1004.62 | 1007.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1010.50 | 1005.80 | 1007.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1010.50 | 1005.80 | 1007.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1010.50 | 1005.80 | 1007.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 1010.50 | 1005.80 | 1007.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1013.05 | 1007.25 | 1008.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:45:00 | 1005.55 | 1007.25 | 1008.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1010.10 | 1007.82 | 1008.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 1001.00 | 1006.81 | 1007.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1016.00 | 1007.36 | 1007.53 | SL hit (close>static) qty=1.00 sl=1014.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 1029.80 | 1007.58 | 1006.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 1030.20 | 1014.76 | 1010.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 15:15:00 | 1015.80 | 1017.83 | 1013.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 15:15:00 | 1015.80 | 1017.83 | 1013.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1015.80 | 1017.83 | 1013.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 1027.50 | 1017.83 | 1013.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 1076.20 | 1089.48 | 1090.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1076.20 | 1089.48 | 1090.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1073.00 | 1086.19 | 1089.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 15:15:00 | 1059.00 | 1057.83 | 1065.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 09:15:00 | 1051.00 | 1057.83 | 1065.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1081.40 | 1058.58 | 1060.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 1081.40 | 1058.58 | 1060.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 1077.20 | 1062.31 | 1062.08 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 1048.60 | 1061.90 | 1062.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1044.10 | 1056.44 | 1059.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1057.60 | 1045.62 | 1051.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1057.60 | 1045.62 | 1051.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1057.60 | 1045.62 | 1051.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1057.60 | 1045.62 | 1051.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1058.00 | 1048.10 | 1051.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1058.00 | 1048.10 | 1051.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 1060.30 | 1054.24 | 1054.11 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 1042.00 | 1052.03 | 1053.34 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 1065.00 | 1053.49 | 1053.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1065.20 | 1057.51 | 1056.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1065.20 | 1065.70 | 1061.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 1065.20 | 1065.70 | 1061.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 1062.80 | 1065.12 | 1061.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 1061.10 | 1065.12 | 1061.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1067.20 | 1065.54 | 1062.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 1060.80 | 1065.54 | 1062.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1062.60 | 1064.95 | 1062.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 1062.10 | 1064.95 | 1062.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 1071.90 | 1066.34 | 1063.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:45:00 | 1064.50 | 1066.34 | 1063.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1082.50 | 1073.74 | 1067.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:00:00 | 1094.50 | 1077.89 | 1070.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 1093.20 | 1099.53 | 1092.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:30:00 | 1093.90 | 1094.02 | 1091.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 1080.40 | 1089.30 | 1089.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 1080.40 | 1089.30 | 1089.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 1076.60 | 1086.76 | 1088.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 1096.70 | 1088.74 | 1089.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 1096.70 | 1088.74 | 1089.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1096.70 | 1088.74 | 1089.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1096.70 | 1088.74 | 1089.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 1099.50 | 1090.90 | 1090.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 1101.50 | 1093.02 | 1091.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 1099.20 | 1105.95 | 1100.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 14:15:00 | 1099.20 | 1105.95 | 1100.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1099.20 | 1105.95 | 1100.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 1099.20 | 1105.95 | 1100.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1101.70 | 1105.10 | 1100.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 1096.60 | 1105.10 | 1100.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1099.70 | 1104.02 | 1100.76 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 1090.60 | 1098.28 | 1098.70 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1109.00 | 1099.52 | 1098.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 1111.00 | 1101.82 | 1099.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 15:15:00 | 1101.10 | 1102.02 | 1100.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:15:00 | 1104.00 | 1102.02 | 1100.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1108.90 | 1103.40 | 1100.84 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 1097.70 | 1103.02 | 1103.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 1068.80 | 1094.44 | 1099.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 1076.70 | 1075.29 | 1083.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 11:45:00 | 1076.00 | 1075.29 | 1083.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1090.30 | 1079.00 | 1081.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:15:00 | 1090.50 | 1079.00 | 1081.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1089.60 | 1081.12 | 1082.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:15:00 | 1095.30 | 1081.12 | 1082.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1100.50 | 1085.00 | 1084.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 1109.50 | 1098.09 | 1091.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 1122.90 | 1124.48 | 1113.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 13:00:00 | 1122.90 | 1124.48 | 1113.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1114.00 | 1120.70 | 1114.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 1107.90 | 1120.70 | 1114.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1102.30 | 1117.02 | 1113.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 1102.30 | 1117.02 | 1113.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1099.80 | 1113.58 | 1112.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 1099.80 | 1113.58 | 1112.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1096.00 | 1110.06 | 1110.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 1088.60 | 1105.77 | 1108.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 1064.00 | 1060.99 | 1070.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:00:00 | 1064.00 | 1060.99 | 1070.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1072.20 | 1063.23 | 1070.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 1070.70 | 1063.23 | 1070.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1070.50 | 1064.68 | 1070.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 1073.40 | 1064.68 | 1070.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1075.60 | 1066.87 | 1071.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 1081.30 | 1066.87 | 1071.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 1084.50 | 1075.44 | 1074.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 13:15:00 | 1085.60 | 1077.47 | 1075.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1073.80 | 1077.88 | 1076.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 1073.80 | 1077.88 | 1076.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1073.80 | 1077.88 | 1076.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 10:45:00 | 1086.60 | 1078.74 | 1076.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 1088.60 | 1078.74 | 1076.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 14:45:00 | 1088.10 | 1082.74 | 1079.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 1070.40 | 1078.31 | 1078.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 1070.40 | 1078.31 | 1078.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1055.40 | 1071.28 | 1075.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1061.90 | 1059.97 | 1066.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 1061.90 | 1059.97 | 1066.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1061.90 | 1059.97 | 1066.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 1061.90 | 1059.97 | 1066.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1066.90 | 1061.35 | 1066.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 1066.90 | 1061.35 | 1066.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1074.00 | 1063.88 | 1067.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1074.00 | 1063.88 | 1067.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1075.00 | 1066.11 | 1068.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 1061.30 | 1066.11 | 1068.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:30:00 | 1069.60 | 1067.31 | 1068.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 1069.90 | 1067.31 | 1068.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 1069.10 | 1067.92 | 1068.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1057.00 | 1049.85 | 1056.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1057.00 | 1049.85 | 1056.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1054.30 | 1050.74 | 1056.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 1051.60 | 1050.74 | 1056.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1058.30 | 1052.25 | 1056.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 1058.30 | 1052.25 | 1056.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1052.40 | 1052.28 | 1056.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 1046.70 | 1050.40 | 1054.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:45:00 | 1049.60 | 1048.79 | 1053.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 11:15:00 | 1047.60 | 1049.27 | 1053.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 14:15:00 | 1062.00 | 1052.01 | 1052.99 | SL hit (close>static) qty=1.00 sl=1059.70 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1064.00 | 1054.41 | 1053.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 1070.00 | 1057.52 | 1055.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1085.40 | 1088.11 | 1075.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:00:00 | 1085.40 | 1088.11 | 1075.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1087.00 | 1094.32 | 1083.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 1088.10 | 1094.32 | 1083.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1083.80 | 1092.22 | 1083.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 1090.70 | 1092.22 | 1083.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1100.40 | 1093.85 | 1085.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 11:30:00 | 1116.30 | 1100.28 | 1089.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 15:15:00 | 1081.90 | 1093.32 | 1093.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 1081.90 | 1093.32 | 1093.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 1073.90 | 1083.52 | 1087.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 10:15:00 | 1097.80 | 1079.82 | 1082.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 10:15:00 | 1097.80 | 1079.82 | 1082.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1097.80 | 1079.82 | 1082.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 1097.80 | 1079.82 | 1082.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 1108.10 | 1085.48 | 1084.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 1114.20 | 1091.22 | 1087.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 1231.80 | 1235.67 | 1223.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 1231.80 | 1235.67 | 1223.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1234.80 | 1234.22 | 1224.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1228.00 | 1234.22 | 1224.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1296.40 | 1284.06 | 1273.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 1307.20 | 1288.58 | 1284.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 1290.50 | 1300.12 | 1300.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 1290.50 | 1300.12 | 1300.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 1278.90 | 1288.53 | 1293.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 1303.90 | 1289.42 | 1292.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1303.90 | 1289.42 | 1292.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1303.90 | 1289.42 | 1292.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1303.90 | 1289.42 | 1292.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1302.60 | 1292.06 | 1293.02 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 1300.50 | 1293.75 | 1293.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 1320.30 | 1300.73 | 1297.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 1306.50 | 1309.00 | 1303.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 1296.10 | 1309.00 | 1303.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1295.40 | 1306.28 | 1303.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 1296.50 | 1306.28 | 1303.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1289.30 | 1302.88 | 1301.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 1288.50 | 1302.88 | 1301.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 1286.10 | 1299.53 | 1300.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 12:15:00 | 1280.30 | 1295.68 | 1298.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 1292.90 | 1291.66 | 1295.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:15:00 | 1278.10 | 1291.66 | 1295.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1279.40 | 1289.20 | 1294.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:00:00 | 1271.90 | 1285.31 | 1291.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 1267.20 | 1281.60 | 1288.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 1271.70 | 1276.34 | 1282.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 1289.70 | 1284.47 | 1284.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 1289.70 | 1284.47 | 1284.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 11:15:00 | 1301.70 | 1287.91 | 1285.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 1313.10 | 1313.84 | 1305.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:45:00 | 1311.80 | 1313.84 | 1305.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 1304.60 | 1310.75 | 1306.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 1304.60 | 1310.75 | 1306.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1303.20 | 1309.24 | 1305.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1303.20 | 1309.24 | 1305.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1304.00 | 1308.19 | 1305.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1311.50 | 1308.19 | 1305.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1305.60 | 1307.64 | 1305.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 1305.60 | 1307.64 | 1305.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1305.50 | 1307.21 | 1305.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 12:15:00 | 1308.80 | 1307.21 | 1305.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 12:45:00 | 1310.00 | 1307.97 | 1306.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 1304.50 | 1315.35 | 1316.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1304.50 | 1315.35 | 1316.64 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 11:15:00 | 1328.00 | 1317.09 | 1316.13 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1306.50 | 1314.49 | 1315.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1286.10 | 1308.81 | 1312.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 1281.80 | 1278.70 | 1289.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 1281.80 | 1278.70 | 1289.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1293.40 | 1281.64 | 1290.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 1293.40 | 1281.64 | 1290.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1286.20 | 1282.55 | 1289.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1275.60 | 1282.10 | 1289.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1297.50 | 1287.62 | 1288.36 | SL hit (close>static) qty=1.00 sl=1294.40 alert=retest2 |

### Cycle 92 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 1299.00 | 1289.89 | 1289.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 1308.80 | 1293.68 | 1291.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 09:15:00 | 1312.20 | 1314.17 | 1305.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 10:00:00 | 1312.20 | 1314.17 | 1305.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1317.30 | 1314.91 | 1309.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:00:00 | 1327.80 | 1317.49 | 1311.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1310.20 | 1337.83 | 1338.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1310.20 | 1337.83 | 1338.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1306.80 | 1327.01 | 1333.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 12:15:00 | 1241.10 | 1238.27 | 1258.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 12:30:00 | 1241.20 | 1238.27 | 1258.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1233.60 | 1208.16 | 1214.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 1232.80 | 1208.16 | 1214.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1227.20 | 1211.96 | 1215.31 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 1231.90 | 1219.05 | 1218.14 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 1216.90 | 1220.50 | 1220.56 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 1232.30 | 1222.67 | 1221.44 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 1209.80 | 1221.99 | 1223.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1194.10 | 1216.41 | 1220.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 11:15:00 | 1196.90 | 1194.04 | 1202.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 1196.90 | 1194.04 | 1202.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1196.90 | 1194.04 | 1202.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 1196.90 | 1194.04 | 1202.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1199.30 | 1195.89 | 1200.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1199.40 | 1195.89 | 1200.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1200.00 | 1196.71 | 1200.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1213.20 | 1196.71 | 1200.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1212.80 | 1199.93 | 1201.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 1218.60 | 1199.93 | 1201.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1213.60 | 1202.66 | 1203.00 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 1207.00 | 1203.53 | 1203.36 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 1199.00 | 1202.84 | 1203.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 10:15:00 | 1197.50 | 1201.05 | 1202.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 1204.40 | 1201.72 | 1202.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 11:15:00 | 1204.40 | 1201.72 | 1202.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1204.40 | 1201.72 | 1202.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 1209.10 | 1201.72 | 1202.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1204.40 | 1202.25 | 1202.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:15:00 | 1206.20 | 1202.25 | 1202.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 1212.50 | 1204.30 | 1203.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 1214.70 | 1206.38 | 1204.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 09:15:00 | 1225.10 | 1230.28 | 1223.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1225.10 | 1230.28 | 1223.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1225.10 | 1230.28 | 1223.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1225.10 | 1230.28 | 1223.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1223.00 | 1228.82 | 1223.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1217.40 | 1228.82 | 1223.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1237.40 | 1230.54 | 1225.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:45:00 | 1229.60 | 1230.54 | 1225.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1240.00 | 1252.35 | 1246.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:15:00 | 1236.60 | 1252.35 | 1246.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1246.10 | 1251.10 | 1246.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 1252.70 | 1250.14 | 1246.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 13:15:00 | 1228.90 | 1245.09 | 1244.69 | SL hit (close<static) qty=1.00 sl=1233.30 alert=retest2 |

### Cycle 101 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 1237.20 | 1243.51 | 1244.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1226.40 | 1238.29 | 1241.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 1239.00 | 1238.43 | 1241.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 1239.00 | 1238.43 | 1241.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1239.00 | 1238.43 | 1241.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 1250.10 | 1238.43 | 1241.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1243.00 | 1239.35 | 1241.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 1243.00 | 1239.35 | 1241.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1249.60 | 1241.40 | 1242.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1249.60 | 1241.40 | 1242.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1221.10 | 1237.34 | 1240.22 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1282.20 | 1243.55 | 1242.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 10:15:00 | 1290.00 | 1252.84 | 1246.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 14:15:00 | 1310.50 | 1311.45 | 1292.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 14:30:00 | 1311.20 | 1311.45 | 1292.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1304.60 | 1317.74 | 1310.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1308.40 | 1317.74 | 1310.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1310.00 | 1316.19 | 1310.26 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1300.70 | 1307.31 | 1307.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1283.40 | 1297.84 | 1301.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 1292.50 | 1289.30 | 1295.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:00:00 | 1292.50 | 1289.30 | 1295.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 1292.70 | 1290.21 | 1295.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 1294.00 | 1290.21 | 1295.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1298.10 | 1291.78 | 1295.41 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 1308.60 | 1298.32 | 1297.50 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 1295.60 | 1297.58 | 1297.76 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 11:15:00 | 1302.80 | 1297.90 | 1297.64 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 1294.80 | 1297.28 | 1297.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 13:15:00 | 1289.30 | 1295.68 | 1296.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1298.70 | 1292.86 | 1294.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1298.70 | 1292.86 | 1294.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1298.70 | 1292.86 | 1294.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 1298.70 | 1292.86 | 1294.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1283.60 | 1291.01 | 1293.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:30:00 | 1283.10 | 1289.81 | 1292.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:00:00 | 1282.80 | 1288.07 | 1291.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 1281.10 | 1288.68 | 1291.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:45:00 | 1282.50 | 1286.11 | 1289.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1288.20 | 1285.68 | 1288.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1288.20 | 1285.68 | 1288.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1298.90 | 1288.61 | 1289.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 1295.90 | 1288.61 | 1289.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 1312.40 | 1293.37 | 1291.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 1312.40 | 1293.37 | 1291.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 1321.10 | 1298.92 | 1293.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 1301.90 | 1310.89 | 1303.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 1301.90 | 1310.89 | 1303.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1301.90 | 1310.89 | 1303.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1301.90 | 1310.89 | 1303.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1297.00 | 1308.11 | 1303.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 1297.00 | 1308.11 | 1303.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1310.50 | 1308.12 | 1304.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 1314.90 | 1308.12 | 1304.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 12:15:00 | 1297.20 | 1304.88 | 1304.32 | SL hit (close<static) qty=1.00 sl=1301.50 alert=retest2 |

### Cycle 109 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1294.80 | 1302.86 | 1303.46 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 1314.00 | 1304.07 | 1303.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 1328.00 | 1308.86 | 1306.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 1310.30 | 1310.45 | 1307.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 11:45:00 | 1313.10 | 1310.45 | 1307.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1312.50 | 1310.86 | 1307.80 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1287.40 | 1305.40 | 1306.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1276.00 | 1293.06 | 1299.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1271.60 | 1271.50 | 1279.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 14:00:00 | 1271.60 | 1271.50 | 1279.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1266.80 | 1267.55 | 1273.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 1266.80 | 1267.55 | 1273.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1271.10 | 1268.26 | 1273.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 1276.50 | 1268.26 | 1273.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1270.40 | 1268.69 | 1273.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:30:00 | 1267.70 | 1268.69 | 1273.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1273.50 | 1269.65 | 1273.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1261.00 | 1269.65 | 1273.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1249.90 | 1241.22 | 1240.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1249.90 | 1241.22 | 1240.62 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 1239.10 | 1240.33 | 1240.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 1233.00 | 1238.41 | 1239.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 11:15:00 | 1242.30 | 1237.90 | 1238.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 1242.30 | 1237.90 | 1238.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1242.30 | 1237.90 | 1238.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 1243.20 | 1237.90 | 1238.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 1264.90 | 1243.30 | 1241.07 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1221.60 | 1242.63 | 1244.13 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1257.60 | 1241.15 | 1239.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1269.00 | 1246.72 | 1242.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 1294.50 | 1297.30 | 1288.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:30:00 | 1292.20 | 1297.30 | 1288.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1290.50 | 1295.94 | 1288.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1290.40 | 1295.94 | 1288.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1278.60 | 1292.47 | 1287.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 1278.60 | 1292.47 | 1287.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1271.40 | 1288.26 | 1286.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1271.40 | 1288.26 | 1286.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1272.60 | 1285.13 | 1285.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 1258.40 | 1267.16 | 1271.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1271.50 | 1267.38 | 1271.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1271.50 | 1267.38 | 1271.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1271.50 | 1267.38 | 1271.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1271.50 | 1267.38 | 1271.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1270.10 | 1267.92 | 1270.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 1265.10 | 1266.12 | 1269.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1267.50 | 1269.31 | 1270.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:00:00 | 1269.30 | 1268.99 | 1270.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 14:15:00 | 1285.80 | 1273.30 | 1272.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1285.80 | 1273.30 | 1272.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1292.20 | 1277.83 | 1274.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1323.10 | 1329.46 | 1316.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:30:00 | 1324.30 | 1329.46 | 1316.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1314.80 | 1326.53 | 1316.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 1320.70 | 1326.53 | 1316.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1315.40 | 1324.31 | 1316.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:30:00 | 1316.90 | 1324.31 | 1316.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1318.80 | 1323.20 | 1316.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 1321.20 | 1323.20 | 1316.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:45:00 | 1321.90 | 1322.54 | 1317.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 1321.70 | 1321.96 | 1318.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 12:15:00 | 1311.00 | 1319.38 | 1317.95 | SL hit (close<static) qty=1.00 sl=1313.10 alert=retest2 |

### Cycle 119 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 1316.20 | 1316.90 | 1316.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1309.40 | 1315.40 | 1316.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 1206.60 | 1201.46 | 1218.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:45:00 | 1206.30 | 1201.46 | 1218.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1208.80 | 1199.81 | 1209.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1208.80 | 1199.81 | 1209.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1202.20 | 1200.28 | 1209.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:15:00 | 1201.40 | 1200.28 | 1209.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 1141.33 | 1163.41 | 1179.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 1133.40 | 1133.21 | 1149.33 | SL hit (close>ema200) qty=0.50 sl=1133.21 alert=retest2 |

### Cycle 120 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 1170.80 | 1153.38 | 1151.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 1178.30 | 1158.37 | 1153.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 1153.00 | 1162.46 | 1158.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 13:15:00 | 1153.00 | 1162.46 | 1158.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1153.00 | 1162.46 | 1158.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1153.00 | 1162.46 | 1158.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1132.30 | 1156.43 | 1155.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 1132.30 | 1156.43 | 1155.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 1136.00 | 1152.34 | 1153.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 1129.50 | 1145.94 | 1150.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1146.10 | 1138.33 | 1144.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1146.10 | 1138.33 | 1144.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1146.10 | 1138.33 | 1144.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1146.10 | 1138.33 | 1144.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1158.40 | 1142.34 | 1146.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:45:00 | 1138.50 | 1139.94 | 1144.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1142.60 | 1144.84 | 1145.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:00:00 | 1142.70 | 1139.02 | 1141.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 1168.80 | 1147.43 | 1144.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 1168.80 | 1147.43 | 1144.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 1178.80 | 1158.60 | 1150.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 1170.80 | 1173.01 | 1163.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 1170.80 | 1173.01 | 1163.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1170.80 | 1173.01 | 1163.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1170.90 | 1173.01 | 1163.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1175.90 | 1173.59 | 1164.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 1169.20 | 1173.59 | 1164.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1143.50 | 1167.57 | 1162.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 1143.50 | 1167.57 | 1162.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1144.00 | 1162.86 | 1160.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 1152.90 | 1160.30 | 1159.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 1142.10 | 1156.66 | 1158.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 1142.10 | 1156.66 | 1158.15 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1174.50 | 1160.81 | 1159.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 1180.00 | 1164.65 | 1161.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 1202.70 | 1208.36 | 1195.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 12:15:00 | 1202.70 | 1208.36 | 1195.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1202.70 | 1208.36 | 1195.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 1202.70 | 1208.36 | 1195.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1183.00 | 1201.66 | 1196.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 1182.20 | 1201.66 | 1196.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1178.10 | 1196.95 | 1194.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 1180.10 | 1196.95 | 1194.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1180.60 | 1209.32 | 1203.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1180.60 | 1209.32 | 1203.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1191.10 | 1205.68 | 1202.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 1172.40 | 1205.68 | 1202.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 1170.70 | 1194.48 | 1197.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 1158.80 | 1187.34 | 1194.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1191.50 | 1179.06 | 1187.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1191.50 | 1179.06 | 1187.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1191.50 | 1179.06 | 1187.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1191.50 | 1179.06 | 1187.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1195.40 | 1182.33 | 1188.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 1197.90 | 1182.33 | 1188.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 1204.00 | 1192.42 | 1192.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 15:15:00 | 1213.60 | 1198.00 | 1194.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1232.80 | 1238.13 | 1227.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 10:00:00 | 1232.80 | 1238.13 | 1227.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1245.00 | 1239.51 | 1228.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 1236.30 | 1239.51 | 1228.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1226.50 | 1241.66 | 1235.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 1226.50 | 1241.66 | 1235.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1231.80 | 1239.69 | 1234.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:30:00 | 1236.30 | 1239.01 | 1235.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 11:15:00 | 1218.30 | 1236.21 | 1236.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 1218.30 | 1236.21 | 1236.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 13:15:00 | 1217.40 | 1230.11 | 1233.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 12:15:00 | 1224.50 | 1222.22 | 1227.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 12:45:00 | 1223.40 | 1222.22 | 1227.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1204.60 | 1208.15 | 1214.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 1198.50 | 1205.81 | 1212.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 14:15:00 | 1199.30 | 1204.74 | 1210.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 1200.30 | 1199.71 | 1205.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 12:00:00 | 1200.10 | 1199.71 | 1205.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1196.60 | 1197.72 | 1202.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1203.20 | 1197.72 | 1202.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1196.10 | 1194.02 | 1198.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 1196.10 | 1194.02 | 1198.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1195.10 | 1183.03 | 1188.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1195.10 | 1183.03 | 1188.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1208.10 | 1188.04 | 1189.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 1208.10 | 1188.04 | 1189.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 1214.40 | 1193.31 | 1192.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1214.40 | 1193.31 | 1192.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1219.60 | 1202.84 | 1197.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1207.10 | 1221.09 | 1212.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1207.10 | 1221.09 | 1212.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1207.10 | 1221.09 | 1212.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1207.10 | 1221.09 | 1212.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1204.30 | 1217.73 | 1212.04 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 1187.00 | 1205.70 | 1207.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1162.70 | 1193.95 | 1201.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 15:15:00 | 1133.00 | 1132.51 | 1150.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:15:00 | 1138.60 | 1132.51 | 1150.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1130.20 | 1120.87 | 1130.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 1130.20 | 1120.87 | 1130.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1133.30 | 1123.36 | 1130.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 1133.30 | 1123.36 | 1130.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1133.00 | 1125.29 | 1130.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 1133.00 | 1125.29 | 1130.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1122.30 | 1124.69 | 1130.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1064.20 | 1125.23 | 1129.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 15:00:00 | 1113.20 | 1102.64 | 1103.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 15:15:00 | 1114.00 | 1104.91 | 1104.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 1114.00 | 1104.91 | 1104.82 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 11:15:00 | 1096.00 | 1103.52 | 1104.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 1086.90 | 1100.20 | 1102.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1019.80 | 1016.39 | 1034.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 1024.90 | 1016.39 | 1034.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1031.80 | 1018.61 | 1032.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1031.80 | 1018.61 | 1032.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1033.70 | 1021.63 | 1032.50 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 1053.50 | 1038.21 | 1037.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1067.20 | 1046.61 | 1041.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1051.70 | 1062.99 | 1054.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1051.70 | 1062.99 | 1054.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1051.70 | 1062.99 | 1054.79 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1038.40 | 1049.01 | 1050.38 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1063.50 | 1051.08 | 1051.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 1069.70 | 1054.80 | 1052.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 15:15:00 | 1058.00 | 1060.86 | 1057.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 09:15:00 | 1029.60 | 1060.86 | 1057.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 135 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1022.50 | 1053.19 | 1053.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 1011.40 | 1036.02 | 1045.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1045.40 | 1035.91 | 1041.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1045.40 | 1035.91 | 1041.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1045.40 | 1035.91 | 1041.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:45:00 | 1047.00 | 1035.91 | 1041.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 1042.10 | 1037.15 | 1041.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 1045.60 | 1037.15 | 1041.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1051.70 | 1040.06 | 1042.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 1054.60 | 1040.06 | 1042.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 1069.20 | 1045.89 | 1045.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 13:15:00 | 1071.90 | 1051.09 | 1047.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1064.40 | 1080.26 | 1070.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1064.40 | 1080.26 | 1070.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1064.40 | 1080.26 | 1070.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 1064.80 | 1080.26 | 1070.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1053.30 | 1074.87 | 1069.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1053.30 | 1074.87 | 1069.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1047.10 | 1065.08 | 1065.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1039.40 | 1054.49 | 1059.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1046.30 | 1041.47 | 1049.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1046.30 | 1041.47 | 1049.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1046.30 | 1041.47 | 1049.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 1044.00 | 1045.60 | 1048.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1044.30 | 1045.60 | 1048.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 15:15:00 | 1052.00 | 1038.65 | 1037.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1052.00 | 1038.65 | 1037.13 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 1024.40 | 1035.80 | 1035.98 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1088.30 | 1038.02 | 1035.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 1097.00 | 1049.82 | 1040.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1062.30 | 1074.65 | 1060.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 1062.30 | 1074.65 | 1060.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1062.30 | 1074.65 | 1060.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 1062.30 | 1074.65 | 1060.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1069.40 | 1073.60 | 1061.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1075.80 | 1064.50 | 1061.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 13:15:00 | 1076.40 | 1065.79 | 1062.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 14:45:00 | 1076.90 | 1070.10 | 1065.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1053.30 | 1067.57 | 1064.89 | SL hit (close<static) qty=1.00 sl=1061.00 alert=retest2 |

### Cycle 141 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 1115.50 | 1130.96 | 1132.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1096.50 | 1124.06 | 1129.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1134.00 | 1115.14 | 1120.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1134.00 | 1115.14 | 1120.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1134.00 | 1115.14 | 1120.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1134.00 | 1115.14 | 1120.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1129.70 | 1118.05 | 1121.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1124.90 | 1118.05 | 1121.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 13:15:00 | 1133.00 | 1124.85 | 1123.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1133.00 | 1124.85 | 1123.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1147.00 | 1131.55 | 1128.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1134.10 | 1136.33 | 1132.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1134.10 | 1136.33 | 1132.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1134.10 | 1136.33 | 1132.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1134.10 | 1136.33 | 1132.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1129.60 | 1134.98 | 1131.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1129.60 | 1134.98 | 1131.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1130.00 | 1133.99 | 1131.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1112.60 | 1133.99 | 1131.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1103.20 | 1127.83 | 1129.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 1100.00 | 1117.97 | 1124.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1129.20 | 1117.08 | 1120.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1129.20 | 1117.08 | 1120.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1129.20 | 1117.08 | 1120.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 1129.20 | 1117.08 | 1120.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1124.30 | 1118.53 | 1121.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 1120.20 | 1116.76 | 1120.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 1127.30 | 1105.75 | 1103.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1127.30 | 1105.75 | 1103.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 1133.00 | 1111.20 | 1106.61 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 15:15:00 | 760.00 | 2024-05-23 15:15:00 | 836.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-23 10:30:00 | 760.95 | 2024-05-23 15:15:00 | 837.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 09:30:00 | 970.75 | 2024-06-18 09:15:00 | 1056.55 | TARGET_HIT | 1.00 | 8.84% |
| BUY | retest2 | 2024-06-11 10:15:00 | 960.50 | 2024-06-18 12:15:00 | 1067.83 | TARGET_HIT | 1.00 | 11.17% |
| SELL | retest2 | 2024-07-16 12:45:00 | 1031.50 | 2024-07-19 09:15:00 | 979.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 12:45:00 | 1031.50 | 2024-07-22 09:15:00 | 974.25 | STOP_HIT | 0.50 | 5.55% |
| BUY | retest2 | 2024-07-26 11:15:00 | 1025.25 | 2024-08-01 09:15:00 | 1038.00 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-08-02 09:15:00 | 1027.00 | 2024-08-05 09:15:00 | 924.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-21 10:30:00 | 1186.15 | 2024-08-22 12:15:00 | 1139.40 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2024-08-28 12:30:00 | 1098.90 | 2024-08-28 13:15:00 | 1120.70 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-08-30 09:15:00 | 1129.20 | 2024-09-02 09:15:00 | 1242.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-30 10:00:00 | 1130.05 | 2024-09-02 09:15:00 | 1243.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-10 10:15:00 | 1001.05 | 2024-10-14 12:15:00 | 1016.55 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-10-10 12:30:00 | 1001.00 | 2024-10-14 12:15:00 | 1016.55 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-10-10 15:00:00 | 1001.10 | 2024-10-14 12:15:00 | 1016.55 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-10-21 12:00:00 | 944.45 | 2024-10-24 12:15:00 | 948.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-10-21 13:30:00 | 944.50 | 2024-10-24 12:15:00 | 948.20 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-10-21 14:00:00 | 944.20 | 2024-10-24 12:15:00 | 948.20 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-10-23 13:45:00 | 944.75 | 2024-10-24 12:15:00 | 948.20 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-10-30 15:15:00 | 969.50 | 2024-11-05 09:15:00 | 955.05 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-10-31 09:30:00 | 967.80 | 2024-11-05 09:15:00 | 955.05 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-11-04 14:30:00 | 966.90 | 2024-11-05 09:15:00 | 955.05 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-11-13 12:00:00 | 968.35 | 2024-11-21 09:15:00 | 1061.89 | TARGET_HIT | 1.00 | 9.66% |
| BUY | retest2 | 2024-11-13 15:15:00 | 965.35 | 2024-11-21 10:15:00 | 1065.19 | TARGET_HIT | 1.00 | 10.34% |
| BUY | retest2 | 2024-11-14 10:15:00 | 973.70 | 2024-11-21 10:15:00 | 1071.07 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-02 09:15:00 | 1038.60 | 2024-12-03 15:15:00 | 1060.50 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-12-09 12:15:00 | 1078.35 | 2024-12-09 12:15:00 | 1074.20 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-12-11 11:15:00 | 1071.10 | 2024-12-11 11:15:00 | 1077.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-12-13 09:15:00 | 1054.90 | 2024-12-16 12:15:00 | 1057.70 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-12-17 10:15:00 | 1038.90 | 2024-12-19 13:15:00 | 1044.65 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-12-27 09:15:00 | 1031.70 | 2024-12-31 13:15:00 | 1050.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-12-30 10:15:00 | 1033.00 | 2024-12-31 13:15:00 | 1050.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-12-30 15:00:00 | 1034.65 | 2024-12-31 13:15:00 | 1050.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-12-31 09:15:00 | 1029.45 | 2024-12-31 13:15:00 | 1050.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-01-03 14:00:00 | 1063.35 | 2025-01-07 12:15:00 | 1052.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-01-03 14:30:00 | 1063.30 | 2025-01-07 12:15:00 | 1052.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-01-06 11:15:00 | 1071.90 | 2025-01-07 12:15:00 | 1052.80 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-01-06 12:30:00 | 1066.50 | 2025-01-07 12:15:00 | 1052.80 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-01-10 11:15:00 | 1080.00 | 2025-01-13 11:15:00 | 1053.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-01-10 13:00:00 | 1078.15 | 2025-01-13 11:15:00 | 1053.50 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-01-10 13:30:00 | 1079.95 | 2025-01-13 11:15:00 | 1053.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-01-10 14:45:00 | 1078.90 | 2025-01-13 11:15:00 | 1053.50 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-01-14 13:30:00 | 1055.00 | 2025-01-16 09:15:00 | 1088.70 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-01-15 10:15:00 | 1050.00 | 2025-01-16 09:15:00 | 1088.70 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-01-15 14:30:00 | 1057.65 | 2025-01-16 09:15:00 | 1088.70 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-01-29 11:15:00 | 896.50 | 2025-01-30 09:15:00 | 911.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-01-29 12:30:00 | 894.95 | 2025-01-30 09:15:00 | 911.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-02-03 13:15:00 | 960.15 | 2025-02-07 09:15:00 | 1056.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-03 13:45:00 | 960.20 | 2025-02-07 09:15:00 | 1056.22 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-21 11:15:00 | 844.45 | 2025-02-28 11:15:00 | 807.17 | PARTIAL | 0.50 | 4.42% |
| SELL | retest2 | 2025-02-21 12:00:00 | 843.00 | 2025-02-28 11:15:00 | 807.60 | PARTIAL | 0.50 | 4.20% |
| SELL | retest2 | 2025-02-21 11:15:00 | 844.45 | 2025-02-28 12:15:00 | 825.00 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2025-02-21 12:00:00 | 843.00 | 2025-02-28 12:15:00 | 825.00 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2025-02-21 13:15:00 | 844.70 | 2025-03-03 09:15:00 | 802.23 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-02-21 14:45:00 | 844.60 | 2025-03-03 09:15:00 | 800.85 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-02-24 14:30:00 | 849.65 | 2025-03-03 09:15:00 | 802.47 | PARTIAL | 0.50 | 5.55% |
| SELL | retest2 | 2025-02-24 15:15:00 | 847.70 | 2025-03-03 09:15:00 | 802.37 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-02-25 10:45:00 | 850.10 | 2025-03-03 09:15:00 | 805.32 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-02-21 13:15:00 | 844.70 | 2025-03-03 12:15:00 | 829.15 | STOP_HIT | 0.50 | 1.84% |
| SELL | retest2 | 2025-02-21 14:45:00 | 844.60 | 2025-03-03 12:15:00 | 829.15 | STOP_HIT | 0.50 | 1.83% |
| SELL | retest2 | 2025-02-24 14:30:00 | 849.65 | 2025-03-03 12:15:00 | 829.15 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2025-02-24 15:15:00 | 847.70 | 2025-03-03 12:15:00 | 829.15 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2025-02-25 10:45:00 | 850.10 | 2025-03-03 12:15:00 | 829.15 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2025-02-27 09:15:00 | 834.00 | 2025-03-04 13:15:00 | 832.50 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-02-27 10:45:00 | 823.25 | 2025-03-04 13:15:00 | 832.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-02-27 14:00:00 | 820.15 | 2025-03-04 13:15:00 | 832.50 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-02-28 09:15:00 | 821.90 | 2025-03-04 13:15:00 | 832.50 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-02-28 10:45:00 | 821.30 | 2025-03-04 13:15:00 | 832.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-03-12 10:15:00 | 846.65 | 2025-03-13 09:15:00 | 873.35 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-03-12 11:15:00 | 850.80 | 2025-03-13 09:15:00 | 873.35 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-03-12 12:15:00 | 852.80 | 2025-03-13 09:15:00 | 873.35 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-03-12 13:45:00 | 852.60 | 2025-03-13 09:15:00 | 873.35 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-03-28 10:15:00 | 912.55 | 2025-03-28 14:15:00 | 866.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 10:15:00 | 912.55 | 2025-04-01 13:15:00 | 897.05 | STOP_HIT | 0.50 | 1.70% |
| BUY | retest2 | 2025-04-17 10:15:00 | 857.05 | 2025-04-25 09:15:00 | 880.80 | STOP_HIT | 1.00 | 2.77% |
| BUY | retest2 | 2025-04-30 11:00:00 | 895.50 | 2025-05-02 10:15:00 | 886.15 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-04-30 12:15:00 | 900.80 | 2025-05-02 10:15:00 | 886.15 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-05-02 09:30:00 | 898.15 | 2025-05-02 10:15:00 | 886.15 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-05-07 11:45:00 | 916.60 | 2025-05-09 12:15:00 | 911.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-05-07 13:30:00 | 917.45 | 2025-05-09 12:15:00 | 911.30 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-05-09 11:30:00 | 917.70 | 2025-05-09 12:15:00 | 911.30 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-05-15 10:15:00 | 994.60 | 2025-05-20 14:15:00 | 980.05 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-16 09:15:00 | 1001.35 | 2025-05-20 14:15:00 | 980.05 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-05-16 10:15:00 | 994.60 | 2025-05-20 14:15:00 | 980.05 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-16 12:45:00 | 994.45 | 2025-05-20 14:15:00 | 980.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-05-23 13:15:00 | 1018.80 | 2025-05-23 14:15:00 | 1006.50 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-05-23 13:45:00 | 1018.45 | 2025-05-23 14:15:00 | 1006.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1023.75 | 2025-05-26 09:15:00 | 1006.45 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-26 10:30:00 | 1019.80 | 2025-05-27 09:15:00 | 1001.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-05-29 12:30:00 | 1001.00 | 2025-05-30 09:15:00 | 1016.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-05-30 10:45:00 | 1005.10 | 2025-06-02 09:15:00 | 1029.80 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-05-30 11:30:00 | 1004.55 | 2025-06-02 09:15:00 | 1029.80 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-05-30 15:15:00 | 1003.50 | 2025-06-02 09:15:00 | 1029.80 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-06-03 09:15:00 | 1027.50 | 2025-06-12 12:15:00 | 1076.20 | STOP_HIT | 1.00 | 4.74% |
| BUY | retest2 | 2025-06-27 11:00:00 | 1094.50 | 2025-07-01 14:15:00 | 1080.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-07-01 10:00:00 | 1093.20 | 2025-07-01 14:15:00 | 1080.40 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-01 12:30:00 | 1093.90 | 2025-07-01 14:15:00 | 1080.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-07-25 10:45:00 | 1086.60 | 2025-07-28 11:15:00 | 1070.40 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-25 11:15:00 | 1088.60 | 2025-07-28 11:15:00 | 1070.40 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-25 14:45:00 | 1088.10 | 2025-07-28 11:15:00 | 1070.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-07-30 09:15:00 | 1061.30 | 2025-08-04 14:15:00 | 1062.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-07-30 12:30:00 | 1069.60 | 2025-08-04 14:15:00 | 1062.00 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-07-30 13:15:00 | 1069.90 | 2025-08-04 14:15:00 | 1062.00 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-07-30 15:00:00 | 1069.10 | 2025-08-04 15:15:00 | 1064.00 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-08-01 14:30:00 | 1046.70 | 2025-08-04 15:15:00 | 1064.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-04 09:45:00 | 1049.60 | 2025-08-04 15:15:00 | 1064.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-04 11:15:00 | 1047.60 | 2025-08-04 15:15:00 | 1064.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-08-07 11:30:00 | 1116.30 | 2025-08-08 15:15:00 | 1081.90 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-09-01 09:30:00 | 1307.20 | 2025-09-03 11:15:00 | 1290.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-09-10 12:00:00 | 1271.90 | 2025-09-15 10:15:00 | 1289.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-10 14:15:00 | 1267.20 | 2025-09-15 10:15:00 | 1289.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-11 12:45:00 | 1271.70 | 2025-09-15 10:15:00 | 1289.70 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-09-18 12:15:00 | 1308.80 | 2025-09-24 11:15:00 | 1304.50 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-09-18 12:45:00 | 1310.00 | 2025-09-24 11:15:00 | 1304.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1275.60 | 2025-09-30 14:15:00 | 1297.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-10-06 11:00:00 | 1327.80 | 2025-10-08 14:15:00 | 1310.20 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-11-06 11:30:00 | 1252.70 | 2025-11-06 13:15:00 | 1228.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-11-24 11:30:00 | 1283.10 | 2025-11-26 10:15:00 | 1312.40 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-11-24 14:00:00 | 1282.80 | 2025-11-26 10:15:00 | 1312.40 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-11-24 15:15:00 | 1281.10 | 2025-11-26 10:15:00 | 1312.40 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-11-25 11:45:00 | 1282.50 | 2025-11-26 10:15:00 | 1312.40 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-11-27 14:15:00 | 1314.90 | 2025-11-28 12:15:00 | 1297.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1261.00 | 2025-12-12 15:15:00 | 1249.90 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1265.10 | 2025-12-31 14:15:00 | 1285.80 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1267.50 | 2025-12-31 14:15:00 | 1285.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-31 13:00:00 | 1269.30 | 2025-12-31 14:15:00 | 1285.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-01-06 13:15:00 | 1321.20 | 2026-01-07 12:15:00 | 1311.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-01-06 14:45:00 | 1321.90 | 2026-01-07 12:15:00 | 1311.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1321.70 | 2026-01-07 12:15:00 | 1311.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-01-07 14:45:00 | 1325.00 | 2026-01-07 15:15:00 | 1316.20 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-01-16 11:15:00 | 1201.40 | 2026-01-20 09:15:00 | 1141.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:15:00 | 1201.40 | 2026-01-21 12:15:00 | 1133.40 | STOP_HIT | 0.50 | 5.66% |
| SELL | retest2 | 2026-01-28 09:45:00 | 1138.50 | 2026-01-30 10:15:00 | 1168.80 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1142.60 | 2026-01-30 10:15:00 | 1168.80 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-01-29 15:00:00 | 1142.70 | 2026-01-30 10:15:00 | 1168.80 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-02-02 09:45:00 | 1152.90 | 2026-02-02 10:15:00 | 1142.10 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-02-13 11:30:00 | 1236.30 | 2026-02-16 11:15:00 | 1218.30 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-19 12:00:00 | 1198.50 | 2026-02-25 11:15:00 | 1214.40 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-02-19 14:15:00 | 1199.30 | 2026-02-25 11:15:00 | 1214.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-20 11:30:00 | 1200.30 | 2026-02-25 11:15:00 | 1214.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-02-20 12:00:00 | 1200.10 | 2026-02-25 11:15:00 | 1214.40 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1064.20 | 2026-03-10 15:15:00 | 1114.00 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2026-03-10 15:00:00 | 1113.20 | 2026-03-10 15:15:00 | 1114.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2026-04-01 13:30:00 | 1044.00 | 2026-04-06 15:15:00 | 1052.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-04-01 14:15:00 | 1044.30 | 2026-04-06 15:15:00 | 1052.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-04-10 09:30:00 | 1075.80 | 2026-04-13 09:15:00 | 1053.30 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-04-10 13:15:00 | 1076.40 | 2026-04-13 09:15:00 | 1053.30 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2026-04-10 14:45:00 | 1076.90 | 2026-04-13 09:15:00 | 1053.30 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-04-13 15:15:00 | 1079.90 | 2026-04-24 09:15:00 | 1115.50 | STOP_HIT | 1.00 | 3.30% |
| BUY | retest2 | 2026-04-15 09:30:00 | 1103.00 | 2026-04-24 09:15:00 | 1115.50 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2026-04-17 12:30:00 | 1095.50 | 2026-04-24 09:15:00 | 1115.50 | STOP_HIT | 1.00 | 1.83% |
| BUY | retest2 | 2026-04-20 10:15:00 | 1095.00 | 2026-04-24 09:15:00 | 1115.50 | STOP_HIT | 1.00 | 1.87% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1124.90 | 2026-04-27 13:15:00 | 1133.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-05-04 11:30:00 | 1120.20 | 2026-05-06 14:15:00 | 1127.30 | STOP_HIT | 1.00 | -0.63% |
