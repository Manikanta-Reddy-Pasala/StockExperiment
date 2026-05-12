# Godrej Industries Ltd. (GODREJIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1202.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 43 |
| PARTIAL | 8 |
| TARGET_HIT | 14 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 27
- **Target hits / Stop hits / Partials:** 14 / 33 / 8
- **Avg / median % per leg:** 0.87% / 0.28%
- **Sum % (uncompounded):** 47.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 15 | 44.1% | 13 | 21 | 0 | 2.24% | 76.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -8.16% | -32.6% |
| BUY @ 3rd Alert (retest2) | 30 | 15 | 50.0% | 13 | 17 | 0 | 3.63% | 108.9% |
| SELL (all) | 21 | 13 | 61.9% | 1 | 12 | 8 | -1.36% | -28.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 13 | 61.9% | 1 | 12 | 8 | -1.36% | -28.5% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -8.16% | -32.6% |
| retest2 (combined) | 51 | 28 | 54.9% | 14 | 29 | 8 | 1.58% | 80.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 09:15:00 | 799.40 | 817.69 | 817.76 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 839.45 | 817.91 | 817.85 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 803.00 | 817.77 | 817.82 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 846.40 | 818.12 | 818.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 851.50 | 819.19 | 818.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 873.15 | 889.73 | 868.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 873.15 | 889.73 | 868.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 12:15:00 | 870.65 | 889.40 | 868.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 13:00:00 | 870.65 | 889.40 | 868.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 870.00 | 888.64 | 868.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:30:00 | 876.45 | 888.50 | 868.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 10:15:00 | 876.45 | 888.50 | 868.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 11:45:00 | 876.70 | 887.51 | 869.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 13:30:00 | 879.45 | 887.32 | 869.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 875.95 | 887.04 | 869.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 878.55 | 887.04 | 869.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 10:45:00 | 878.95 | 886.93 | 869.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:30:00 | 878.40 | 887.80 | 871.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-27 09:15:00 | 964.10 | 905.28 | 885.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 982.50 | 1049.84 | 1049.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 970.50 | 1047.70 | 1048.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 10:15:00 | 1026.20 | 1025.74 | 1036.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 10:15:00 | 1026.20 | 1025.74 | 1036.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1026.20 | 1025.74 | 1036.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 1026.20 | 1025.74 | 1036.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1028.40 | 1023.47 | 1034.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:30:00 | 1027.05 | 1023.47 | 1034.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1042.25 | 1023.70 | 1034.66 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 1066.20 | 1042.77 | 1042.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 1082.65 | 1043.83 | 1043.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1108.00 | 1111.07 | 1085.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 1108.00 | 1111.07 | 1085.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 1092.05 | 1111.10 | 1085.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:30:00 | 1108.05 | 1110.99 | 1086.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 12:15:00 | 1084.50 | 1110.30 | 1086.05 | SL hit (close<static) qty=1.00 sl=1085.80 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 09:15:00 | 964.30 | 1067.43 | 1067.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 10:15:00 | 958.35 | 1066.34 | 1067.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 10:15:00 | 911.05 | 893.26 | 946.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 11:00:00 | 911.05 | 893.26 | 946.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 953.25 | 893.85 | 946.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:00:00 | 953.25 | 893.85 | 946.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 1011.90 | 895.03 | 946.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 1011.90 | 895.03 | 946.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 1003.00 | 896.10 | 947.03 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 1102.00 | 985.59 | 985.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 12:15:00 | 1113.90 | 998.18 | 991.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 1086.75 | 1101.30 | 1057.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:45:00 | 1119.05 | 1100.45 | 1059.25 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 11:15:00 | 1156.50 | 1100.45 | 1059.25 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 14:00:00 | 1118.00 | 1100.82 | 1060.06 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 10:00:00 | 1123.95 | 1101.59 | 1061.05 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 1090.00 | 1102.06 | 1063.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 1057.30 | 1102.06 | 1063.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1037.00 | 1101.41 | 1063.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1037.00 | 1101.41 | 1063.72 | SL hit (close<ema400) qty=1.00 sl=1063.72 alert=retest1 |

### Cycle 9 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 1143.00 | 1176.65 | 1176.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 1139.20 | 1176.27 | 1176.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 1203.40 | 1137.86 | 1152.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 13:15:00 | 1203.40 | 1137.86 | 1152.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1203.40 | 1137.86 | 1152.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1203.40 | 1137.86 | 1152.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1169.10 | 1138.17 | 1152.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 1157.00 | 1139.32 | 1152.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 1226.50 | 1141.95 | 1153.89 | SL hit (close>static) qty=1.00 sl=1220.10 alert=retest2 |

### Cycle 10 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1285.00 | 1164.97 | 1164.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 1299.00 | 1175.51 | 1170.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1198.40 | 1202.73 | 1186.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 1198.40 | 1202.73 | 1186.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1185.10 | 1201.95 | 1187.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1185.10 | 1201.95 | 1187.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1188.20 | 1201.81 | 1187.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 1204.70 | 1201.54 | 1187.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 1184.00 | 1201.26 | 1187.57 | SL hit (close<static) qty=1.00 sl=1185.10 alert=retest2 |

### Cycle 11 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1115.10 | 1188.82 | 1188.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1107.60 | 1188.01 | 1188.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 12:15:00 | 1122.20 | 1118.41 | 1143.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 12:30:00 | 1125.80 | 1118.41 | 1143.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1055.30 | 1020.46 | 1047.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 1055.30 | 1020.46 | 1047.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1050.50 | 1020.75 | 1047.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1045.50 | 1020.75 | 1047.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1034.10 | 1021.19 | 1047.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1030.10 | 1021.19 | 1047.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 1026.10 | 1021.63 | 1046.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 978.59 | 1014.50 | 1037.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:15:00 | 974.79 | 1014.15 | 1036.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1001.00 | 998.61 | 1021.96 | SL hit (close>ema200) qty=0.50 sl=998.61 alert=retest2 |

### Cycle 12 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 1155.05 | 940.22 | 939.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 1212.00 | 942.92 | 940.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-08-07 09:30:00 | 876.45 | 2024-08-27 09:15:00 | 964.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-07 10:15:00 | 876.45 | 2024-08-27 09:15:00 | 964.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-08 11:45:00 | 876.70 | 2024-08-27 09:15:00 | 964.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-08 13:30:00 | 879.45 | 2024-08-27 09:15:00 | 967.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-09 09:15:00 | 878.55 | 2024-08-27 09:15:00 | 966.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-09 10:45:00 | 878.95 | 2024-08-27 09:15:00 | 966.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 09:30:00 | 878.40 | 2024-08-27 09:15:00 | 966.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-07 09:30:00 | 1108.05 | 2025-01-07 12:15:00 | 1084.50 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2025-04-02 10:45:00 | 1119.05 | 2025-04-07 09:15:00 | 1037.00 | STOP_HIT | 1.00 | -7.33% |
| BUY | retest1 | 2025-04-02 11:15:00 | 1156.50 | 2025-04-07 09:15:00 | 1037.00 | STOP_HIT | 1.00 | -10.33% |
| BUY | retest1 | 2025-04-02 14:00:00 | 1118.00 | 2025-04-07 09:15:00 | 1037.00 | STOP_HIT | 1.00 | -7.25% |
| BUY | retest1 | 2025-04-03 10:00:00 | 1123.95 | 2025-04-07 09:15:00 | 1037.00 | STOP_HIT | 1.00 | -7.74% |
| BUY | retest2 | 2025-04-08 09:15:00 | 1081.35 | 2025-04-22 11:15:00 | 1182.50 | TARGET_HIT | 1.00 | 9.35% |
| BUY | retest2 | 2025-04-08 13:00:00 | 1075.00 | 2025-04-22 11:15:00 | 1183.60 | TARGET_HIT | 1.00 | 10.10% |
| BUY | retest2 | 2025-04-08 13:45:00 | 1076.00 | 2025-04-22 14:15:00 | 1189.48 | TARGET_HIT | 1.00 | 10.55% |
| BUY | retest2 | 2025-05-02 10:00:00 | 1077.50 | 2025-05-21 09:15:00 | 1185.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-06 12:00:00 | 1104.80 | 2025-06-03 11:15:00 | 1215.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 1117.00 | 2025-06-06 09:15:00 | 1228.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-14 13:15:00 | 1100.80 | 2025-07-17 12:15:00 | 1136.40 | STOP_HIT | 1.00 | 3.23% |
| BUY | retest2 | 2025-07-15 09:15:00 | 1100.00 | 2025-07-17 12:15:00 | 1136.40 | STOP_HIT | 1.00 | 3.31% |
| BUY | retest2 | 2025-07-16 13:15:00 | 1153.70 | 2025-07-21 13:15:00 | 1138.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-17 11:15:00 | 1152.30 | 2025-07-22 15:15:00 | 1143.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-18 14:45:00 | 1153.00 | 2025-07-22 15:15:00 | 1143.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-08-14 12:00:00 | 1157.00 | 2025-08-18 10:15:00 | 1226.50 | STOP_HIT | 1.00 | -6.01% |
| BUY | retest2 | 2025-09-04 15:15:00 | 1204.70 | 2025-09-05 11:15:00 | 1184.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-09-05 13:30:00 | 1197.70 | 2025-09-19 12:15:00 | 1181.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-22 10:15:00 | 1192.90 | 2025-09-22 11:15:00 | 1184.90 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-24 09:45:00 | 1192.40 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-09-25 10:45:00 | 1213.40 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-09-26 10:15:00 | 1201.70 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-09-26 12:30:00 | 1197.00 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-09-26 15:15:00 | 1199.00 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-09-30 09:45:00 | 1201.00 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-09-30 15:00:00 | 1202.30 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-10-01 11:15:00 | 1201.10 | 2025-10-03 10:15:00 | 1172.90 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1030.10 | 2026-01-21 10:15:00 | 978.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1026.10 | 2026-01-21 11:15:00 | 974.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1030.10 | 2026-02-03 09:15:00 | 1001.00 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1026.10 | 2026-02-03 09:15:00 | 1001.00 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2026-02-03 13:30:00 | 1024.40 | 2026-02-13 13:15:00 | 980.88 | PARTIAL | 0.50 | 4.25% |
| SELL | retest2 | 2026-02-10 10:00:00 | 1032.50 | 2026-02-13 14:15:00 | 973.18 | PARTIAL | 0.50 | 5.75% |
| SELL | retest2 | 2026-02-12 09:15:00 | 997.30 | 2026-02-13 14:15:00 | 962.25 | PARTIAL | 0.50 | 3.51% |
| SELL | retest2 | 2026-02-12 12:30:00 | 1011.30 | 2026-02-13 15:15:00 | 960.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 13:15:00 | 1012.90 | 2026-02-13 15:15:00 | 960.50 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-02-03 13:30:00 | 1024.40 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2026-02-10 10:00:00 | 1032.50 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2026-02-12 09:15:00 | 997.30 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -2.43% |
| SELL | retest2 | 2026-02-12 12:30:00 | 1011.30 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -1.01% |
| SELL | retest2 | 2026-02-12 13:15:00 | 1012.90 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -0.85% |
| SELL | retest2 | 2026-02-12 14:00:00 | 1011.05 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-03-02 09:15:00 | 977.60 | 2026-03-05 11:15:00 | 928.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 977.60 | 2026-03-12 09:15:00 | 879.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 968.00 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -24.81% |
| SELL | retest2 | 2026-04-24 13:00:00 | 977.10 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -23.65% |
| SELL | retest2 | 2026-04-29 09:45:00 | 974.25 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -24.01% |
