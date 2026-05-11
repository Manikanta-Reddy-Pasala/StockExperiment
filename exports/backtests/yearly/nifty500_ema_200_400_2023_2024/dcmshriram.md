# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 1237.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 19 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT2_SKIP | 7 |
| ALERT3 | 68 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 13 |
| TARGET_HIT | 13 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 43
- **Target hits / Stop hits / Partials:** 13 / 46 / 13
- **Avg / median % per leg:** 1.30% / -1.28%
- **Sum % (uncompounded):** 93.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 7 | 30.4% | 5 | 18 | 0 | 1.12% | 25.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 7 | 30.4% | 5 | 18 | 0 | 1.12% | 25.8% |
| SELL (all) | 49 | 22 | 44.9% | 8 | 28 | 13 | 1.39% | 68.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 49 | 22 | 44.9% | 8 | 28 | 13 | 1.39% | 68.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 29 | 40.3% | 13 | 46 | 13 | 1.30% | 93.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 13:15:00 | 860.00 | 836.00 | 835.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 13:15:00 | 862.15 | 839.12 | 837.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 15:15:00 | 890.00 | 891.63 | 876.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-25 09:15:00 | 865.80 | 891.63 | 876.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 860.00 | 891.32 | 876.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 11:30:00 | 875.95 | 868.10 | 867.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-31 09:15:00 | 963.55 | 891.76 | 881.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 12:15:00 | 889.00 | 970.20 | 970.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 14:15:00 | 883.05 | 968.53 | 969.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 957.40 | 929.74 | 945.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 957.40 | 929.74 | 945.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 957.40 | 929.74 | 945.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:30:00 | 956.05 | 929.74 | 945.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 950.35 | 929.95 | 945.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:45:00 | 954.45 | 929.95 | 945.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 948.00 | 930.28 | 945.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:45:00 | 947.25 | 930.28 | 945.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 944.20 | 930.42 | 945.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:30:00 | 946.45 | 930.42 | 945.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 938.45 | 930.50 | 945.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 14:30:00 | 944.85 | 930.50 | 945.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 947.05 | 930.71 | 945.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:00:00 | 947.05 | 930.71 | 945.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 947.30 | 930.88 | 945.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 947.30 | 930.88 | 945.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 943.75 | 931.01 | 945.67 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 11:15:00 | 976.95 | 954.74 | 954.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 09:15:00 | 986.45 | 955.85 | 955.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 1019.10 | 1026.12 | 1003.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 11:30:00 | 1017.30 | 1026.12 | 1003.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 14:15:00 | 986.30 | 1024.72 | 1003.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 14:45:00 | 987.00 | 1024.72 | 1003.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 15:15:00 | 987.00 | 1024.34 | 1003.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 09:15:00 | 994.55 | 1024.34 | 1003.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:45:00 | 993.65 | 1023.65 | 1003.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 12:45:00 | 994.20 | 1019.98 | 1004.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 15:00:00 | 998.95 | 1013.10 | 1005.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 984.10 | 1012.76 | 1005.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 10:00:00 | 984.10 | 1012.76 | 1005.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 983.20 | 1012.47 | 1005.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 14:00:00 | 1000.45 | 1011.68 | 1005.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 15:15:00 | 991.00 | 1011.35 | 1004.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-13 11:15:00 | 976.10 | 1010.26 | 1004.48 | SL hit (close<static) qty=1.00 sl=976.50 alert=retest2 |

### Cycle 4 — SELL (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 10:15:00 | 973.30 | 999.57 | 999.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 11:15:00 | 970.70 | 999.29 | 999.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 12:15:00 | 1016.00 | 994.65 | 996.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 12:15:00 | 1016.00 | 994.65 | 996.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 1016.00 | 994.65 | 996.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:00:00 | 1016.00 | 994.65 | 996.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 994.75 | 994.65 | 996.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 14:30:00 | 989.00 | 994.75 | 997.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 09:15:00 | 987.00 | 994.77 | 997.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 09:45:00 | 988.10 | 994.75 | 996.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:45:00 | 983.15 | 994.68 | 996.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 14:15:00 | 939.55 | 983.60 | 990.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 14:15:00 | 938.69 | 983.60 | 990.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 13:15:00 | 937.65 | 981.43 | 989.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 10:15:00 | 933.99 | 979.94 | 988.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 09:15:00 | 890.10 | 971.79 | 983.54 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 10:15:00 | 991.95 | 949.43 | 949.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 12:15:00 | 1004.40 | 957.75 | 953.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 953.00 | 974.80 | 965.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 953.00 | 974.80 | 965.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 953.00 | 974.80 | 965.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 947.25 | 974.80 | 965.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 936.65 | 974.42 | 965.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 936.65 | 974.42 | 965.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 945.40 | 967.26 | 962.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 953.70 | 965.76 | 961.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-13 12:15:00 | 1049.07 | 974.22 | 966.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 11:15:00 | 988.15 | 1072.04 | 1072.10 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 15:15:00 | 1273.25 | 1067.55 | 1066.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 09:15:00 | 1288.55 | 1069.75 | 1067.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 1153.70 | 1169.96 | 1126.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:00:00 | 1153.70 | 1169.96 | 1126.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 1131.20 | 1167.95 | 1128.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 1129.95 | 1167.95 | 1128.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1133.75 | 1166.70 | 1128.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 1130.05 | 1166.70 | 1128.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1122.20 | 1165.95 | 1128.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 1122.20 | 1165.95 | 1128.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1120.35 | 1165.50 | 1128.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 1120.35 | 1165.50 | 1128.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1117.00 | 1165.01 | 1128.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:15:00 | 1124.75 | 1165.01 | 1128.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1110.95 | 1163.50 | 1128.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:15:00 | 1118.25 | 1163.50 | 1128.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 10:15:00 | 1116.00 | 1157.28 | 1129.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:00:00 | 1116.90 | 1156.88 | 1129.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:30:00 | 1116.65 | 1156.56 | 1129.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1112.35 | 1154.37 | 1128.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-06 14:15:00 | 1096.90 | 1152.48 | 1128.44 | SL hit (close<static) qty=1.00 sl=1105.05 alert=retest2 |

### Cycle 8 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 1079.00 | 1113.61 | 1113.69 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 12:15:00 | 1147.65 | 1113.32 | 1113.27 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1073.70 | 1113.20 | 1113.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1070.95 | 1112.78 | 1113.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 14:15:00 | 1104.95 | 1088.40 | 1098.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 14:15:00 | 1104.95 | 1088.40 | 1098.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1104.95 | 1088.40 | 1098.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 15:00:00 | 1104.95 | 1088.40 | 1098.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1105.00 | 1088.56 | 1098.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 1102.00 | 1088.56 | 1098.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1118.50 | 1089.18 | 1099.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:45:00 | 1123.15 | 1089.18 | 1099.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1117.70 | 1091.02 | 1099.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:45:00 | 1119.95 | 1091.02 | 1099.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1101.65 | 1092.32 | 1100.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:00:00 | 1094.00 | 1099.52 | 1103.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 14:15:00 | 1201.70 | 1100.08 | 1103.27 | SL hit (close>static) qty=1.00 sl=1111.25 alert=retest2 |

### Cycle 11 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1176.80 | 1106.43 | 1106.38 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 1074.05 | 1107.69 | 1107.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 1050.10 | 1105.24 | 1106.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 1032.35 | 1030.99 | 1060.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 11:30:00 | 1033.05 | 1030.99 | 1060.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1056.20 | 1031.76 | 1059.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:30:00 | 1074.50 | 1031.76 | 1059.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1057.80 | 1032.02 | 1059.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:00:00 | 1057.80 | 1032.02 | 1059.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1051.30 | 1032.21 | 1059.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:30:00 | 1057.70 | 1032.21 | 1059.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 1059.95 | 1024.86 | 1048.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:00:00 | 1059.95 | 1024.86 | 1048.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 1062.85 | 1025.24 | 1048.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:45:00 | 1062.70 | 1025.24 | 1048.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1062.85 | 1039.14 | 1053.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:00:00 | 1049.25 | 1039.24 | 1053.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 14:15:00 | 996.79 | 1040.32 | 1053.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 1123.20 | 1040.32 | 1053.16 | SL hit (close>static) qty=0.50 sl=1040.32 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1072.90 | 1047.50 | 1047.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 1081.10 | 1047.83 | 1047.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 1048.90 | 1049.67 | 1048.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 14:15:00 | 1048.90 | 1049.67 | 1048.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1048.90 | 1049.67 | 1048.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 1047.20 | 1049.67 | 1048.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1050.00 | 1049.67 | 1048.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1057.10 | 1049.67 | 1048.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:30:00 | 1054.70 | 1050.22 | 1048.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1066.90 | 1050.25 | 1048.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-18 09:15:00 | 1162.81 | 1077.83 | 1064.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 1194.10 | 1263.01 | 1263.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 1181.60 | 1259.54 | 1261.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 1211.00 | 1209.10 | 1229.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:15:00 | 1202.50 | 1209.10 | 1229.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1221.10 | 1208.88 | 1228.63 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 1275.50 | 1241.05 | 1241.00 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 1211.60 | 1241.24 | 1241.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1197.00 | 1239.92 | 1240.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 10:15:00 | 1242.40 | 1229.36 | 1235.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 1242.40 | 1229.36 | 1235.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1242.40 | 1229.36 | 1235.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 1242.40 | 1229.36 | 1235.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1257.00 | 1229.64 | 1235.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 1255.50 | 1229.64 | 1235.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1266.50 | 1226.21 | 1232.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 1266.50 | 1226.21 | 1232.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 1246.70 | 1226.42 | 1232.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 1226.70 | 1226.42 | 1232.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1290.10 | 1224.18 | 1229.57 | SL hit (close>static) qty=1.00 sl=1274.70 alert=retest2 |

### Cycle 17 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1259.40 | 1233.86 | 1233.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1261.10 | 1236.90 | 1235.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 1237.20 | 1241.32 | 1237.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 11:15:00 | 1237.20 | 1241.32 | 1237.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1237.20 | 1241.32 | 1237.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 1237.20 | 1241.32 | 1237.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1237.80 | 1241.28 | 1237.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:45:00 | 1235.50 | 1241.28 | 1237.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1226.40 | 1241.13 | 1237.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 1226.40 | 1241.13 | 1237.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1229.80 | 1241.02 | 1237.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 1233.80 | 1240.84 | 1237.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 13:45:00 | 1233.10 | 1240.55 | 1237.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 1222.60 | 1240.37 | 1237.60 | SL hit (close<static) qty=1.00 sl=1226.40 alert=retest2 |

### Cycle 18 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 1180.00 | 1234.75 | 1234.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 1162.80 | 1220.48 | 1227.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 1179.20 | 1178.73 | 1201.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-29 13:00:00 | 1179.20 | 1178.73 | 1201.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1173.40 | 1175.89 | 1197.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 1162.60 | 1175.65 | 1196.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:45:00 | 1154.90 | 1175.42 | 1196.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 13:00:00 | 1163.10 | 1175.12 | 1196.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 13:30:00 | 1163.10 | 1174.99 | 1196.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1104.47 | 1163.13 | 1184.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1097.15 | 1163.13 | 1184.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1104.94 | 1163.13 | 1184.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1104.94 | 1163.13 | 1184.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-27 10:15:00 | 1046.34 | 1127.31 | 1158.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 19 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 1200.00 | 1119.85 | 1119.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 09:15:00 | 1212.00 | 1149.65 | 1136.51 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-09 11:30:00 | 875.95 | 2023-08-31 09:15:00 | 963.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-02 09:15:00 | 877.70 | 2023-11-07 12:15:00 | 889.00 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2023-11-02 13:00:00 | 874.10 | 2023-11-07 12:15:00 | 889.00 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2024-01-18 09:15:00 | 994.55 | 2024-02-13 11:15:00 | 976.10 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-01-18 10:45:00 | 993.65 | 2024-02-13 11:15:00 | 976.10 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-01-24 12:45:00 | 994.20 | 2024-02-13 11:15:00 | 976.10 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-02-09 15:00:00 | 998.95 | 2024-02-13 11:15:00 | 976.10 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-02-12 14:00:00 | 1000.45 | 2024-02-13 15:15:00 | 975.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2024-02-12 15:15:00 | 991.00 | 2024-02-13 15:15:00 | 975.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-02-15 10:00:00 | 988.90 | 2024-02-20 11:15:00 | 972.25 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-02-15 11:45:00 | 988.95 | 2024-02-20 11:15:00 | 972.25 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-02-27 14:30:00 | 989.00 | 2024-03-06 14:15:00 | 939.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-28 09:15:00 | 987.00 | 2024-03-06 14:15:00 | 938.69 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2024-02-28 09:45:00 | 988.10 | 2024-03-07 13:15:00 | 937.65 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2024-02-28 10:45:00 | 983.15 | 2024-03-11 10:15:00 | 933.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-27 14:30:00 | 989.00 | 2024-03-13 09:15:00 | 890.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-28 09:15:00 | 987.00 | 2024-03-13 09:15:00 | 888.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-28 09:45:00 | 988.10 | 2024-03-13 09:15:00 | 889.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-28 10:45:00 | 983.15 | 2024-03-13 10:15:00 | 884.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-08 10:45:00 | 935.05 | 2024-04-09 15:15:00 | 961.00 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-04-09 09:30:00 | 937.20 | 2024-04-09 15:15:00 | 961.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-04-09 10:15:00 | 937.55 | 2024-04-09 15:15:00 | 961.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-04-15 09:15:00 | 927.50 | 2024-04-26 10:15:00 | 965.05 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-06-07 09:15:00 | 953.70 | 2024-06-13 12:15:00 | 1049.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 11:15:00 | 1118.25 | 2024-12-06 14:15:00 | 1096.90 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-12-05 10:15:00 | 1116.00 | 2024-12-06 14:15:00 | 1096.90 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-12-05 11:00:00 | 1116.90 | 2024-12-06 14:15:00 | 1096.90 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-05 11:30:00 | 1116.65 | 2024-12-06 14:15:00 | 1096.90 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-12-10 10:30:00 | 1123.85 | 2024-12-10 14:15:00 | 1104.10 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-12-11 09:15:00 | 1121.70 | 2024-12-11 12:15:00 | 1107.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-01-27 10:00:00 | 1094.00 | 2025-01-27 14:15:00 | 1201.70 | STOP_HIT | 1.00 | -9.84% |
| SELL | retest2 | 2025-03-26 11:00:00 | 1049.25 | 2025-03-27 14:15:00 | 996.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 11:00:00 | 1049.25 | 2025-03-27 14:15:00 | 1123.20 | STOP_HIT | 0.50 | -7.05% |
| SELL | retest2 | 2025-04-01 12:45:00 | 1050.50 | 2025-04-02 09:15:00 | 1073.10 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-04-04 14:30:00 | 1037.95 | 2025-04-07 09:15:00 | 986.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1012.20 | 2025-04-07 09:15:00 | 961.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 14:30:00 | 1037.95 | 2025-04-15 11:15:00 | 1047.30 | STOP_HIT | 0.50 | -0.90% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1012.20 | 2025-04-15 11:15:00 | 1047.30 | STOP_HIT | 0.50 | -3.47% |
| SELL | retest2 | 2025-04-25 09:45:00 | 1035.90 | 2025-05-02 09:15:00 | 984.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:45:00 | 1035.90 | 2025-05-06 09:15:00 | 1090.00 | STOP_HIT | 0.50 | -5.22% |
| SELL | retest2 | 2025-05-06 13:30:00 | 1036.30 | 2025-05-06 15:15:00 | 984.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 13:30:00 | 1036.30 | 2025-05-12 13:15:00 | 1031.15 | STOP_HIT | 0.50 | 0.50% |
| SELL | retest2 | 2025-05-06 15:00:00 | 1012.90 | 2025-05-19 10:15:00 | 1055.90 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2025-05-13 12:30:00 | 1035.50 | 2025-05-19 10:15:00 | 1055.90 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-05-15 15:15:00 | 1037.30 | 2025-05-19 10:15:00 | 1055.90 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-05-16 14:30:00 | 1037.35 | 2025-05-19 10:15:00 | 1055.90 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-05-26 12:00:00 | 1030.00 | 2025-05-27 09:15:00 | 1053.30 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-05-26 13:45:00 | 1035.75 | 2025-05-27 09:15:00 | 1053.30 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1057.10 | 2025-06-18 09:15:00 | 1162.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-05 14:30:00 | 1054.70 | 2025-06-18 09:15:00 | 1160.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1066.90 | 2025-06-18 09:15:00 | 1173.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-27 09:15:00 | 1226.70 | 2025-12-11 09:15:00 | 1290.10 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest2 | 2025-12-12 15:00:00 | 1244.50 | 2025-12-22 13:15:00 | 1245.90 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-12-16 11:00:00 | 1242.20 | 2025-12-22 13:15:00 | 1245.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-16 12:00:00 | 1243.70 | 2025-12-22 13:15:00 | 1245.90 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-12-17 09:15:00 | 1234.90 | 2025-12-26 12:15:00 | 1259.40 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-12-17 13:15:00 | 1234.20 | 2025-12-26 12:15:00 | 1259.40 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-22 10:00:00 | 1234.60 | 2025-12-26 12:15:00 | 1259.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1233.80 | 2026-01-07 14:15:00 | 1222.60 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-07 13:45:00 | 1233.10 | 2026-01-07 14:15:00 | 1222.60 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1162.60 | 2026-02-16 09:15:00 | 1104.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 09:45:00 | 1154.90 | 2026-02-16 09:15:00 | 1097.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 13:00:00 | 1163.10 | 2026-02-16 09:15:00 | 1104.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 13:30:00 | 1163.10 | 2026-02-16 09:15:00 | 1104.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1162.60 | 2026-02-27 10:15:00 | 1046.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 09:45:00 | 1154.90 | 2026-02-27 10:15:00 | 1039.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 13:00:00 | 1163.10 | 2026-02-27 10:15:00 | 1046.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 13:30:00 | 1163.10 | 2026-02-27 10:15:00 | 1046.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-06 13:30:00 | 1114.80 | 2026-04-06 14:15:00 | 1132.40 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-04-08 14:15:00 | 1118.50 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2026-04-08 15:15:00 | 1117.00 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2026-04-13 11:00:00 | 1120.80 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.05% |
