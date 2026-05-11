# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 1294.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 208 |
| ALERT1 | 138 |
| ALERT2 | 136 |
| ALERT2_SKIP | 83 |
| ALERT3 | 406 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 189 |
| PARTIAL | 8 |
| TARGET_HIT | 11 |
| STOP_HIT | 185 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 203 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 140
- **Target hits / Stop hits / Partials:** 11 / 184 / 8
- **Avg / median % per leg:** 0.26% / -0.62%
- **Sum % (uncompounded):** 51.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 111 | 39 | 35.1% | 10 | 101 | 0 | 0.58% | 63.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.98% | -3.9% |
| BUY @ 3rd Alert (retest2) | 107 | 39 | 36.4% | 10 | 97 | 0 | 0.63% | 67.8% |
| SELL (all) | 92 | 24 | 26.1% | 1 | 83 | 8 | -0.13% | -12.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.10% | -4.2% |
| SELL @ 3rd Alert (retest2) | 90 | 24 | 26.7% | 1 | 81 | 8 | -0.09% | -7.8% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.35% | -8.1% |
| retest2 (combined) | 197 | 63 | 32.0% | 11 | 178 | 8 | 0.30% | 60.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 15:15:00 | 891.00 | 889.58 | 889.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 893.81 | 890.43 | 889.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 10:15:00 | 896.88 | 900.10 | 897.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 10:15:00 | 896.88 | 900.10 | 897.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 896.88 | 900.10 | 897.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:00:00 | 896.88 | 900.10 | 897.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 899.51 | 899.98 | 897.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 12:15:00 | 900.61 | 899.98 | 897.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 15:00:00 | 901.78 | 899.76 | 897.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 15:15:00 | 901.88 | 904.91 | 905.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 15:15:00 | 901.88 | 904.91 | 905.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 11:15:00 | 898.20 | 902.91 | 904.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 909.67 | 903.13 | 903.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 909.67 | 903.13 | 903.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 909.67 | 903.13 | 903.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 10:00:00 | 909.67 | 903.13 | 903.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-06-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 10:15:00 | 909.58 | 904.42 | 904.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 14:15:00 | 910.47 | 906.82 | 905.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 12:15:00 | 922.41 | 922.45 | 917.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-05 12:30:00 | 922.35 | 922.45 | 917.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 929.29 | 932.64 | 929.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:00:00 | 929.29 | 932.64 | 929.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 929.99 | 932.11 | 929.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-08 15:15:00 | 931.70 | 931.61 | 929.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 15:00:00 | 934.19 | 933.83 | 932.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-30 09:15:00 | 1024.87 | 1020.31 | 1012.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 10:15:00 | 1035.14 | 1038.92 | 1039.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 13:15:00 | 1030.59 | 1036.67 | 1037.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 1039.50 | 1035.27 | 1036.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 1039.50 | 1035.27 | 1036.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1039.50 | 1035.27 | 1036.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:45:00 | 1040.88 | 1035.27 | 1036.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 1038.38 | 1035.89 | 1036.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:30:00 | 1040.39 | 1035.89 | 1036.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 1036.22 | 1034.58 | 1035.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:00:00 | 1036.22 | 1034.58 | 1035.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 1036.81 | 1035.02 | 1035.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 11:00:00 | 1036.81 | 1035.02 | 1035.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 11:15:00 | 1034.96 | 1035.01 | 1035.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 12:15:00 | 1033.12 | 1035.01 | 1035.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 13:15:00 | 1038.20 | 1035.62 | 1035.95 | SL hit (close>static) qty=1.00 sl=1036.98 alert=retest2 |

### Cycle 5 — BUY (started 2023-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 15:15:00 | 1038.06 | 1036.40 | 1036.27 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-07-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 11:15:00 | 1033.27 | 1035.98 | 1036.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 1028.31 | 1034.13 | 1035.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 1036.83 | 1025.77 | 1028.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 1036.83 | 1025.77 | 1028.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 1036.83 | 1025.77 | 1028.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:30:00 | 1037.40 | 1025.77 | 1028.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 1033.40 | 1027.30 | 1029.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:15:00 | 1041.94 | 1027.30 | 1029.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 12:15:00 | 1043.35 | 1032.76 | 1031.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 09:15:00 | 1066.98 | 1046.38 | 1041.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 10:15:00 | 1061.60 | 1062.84 | 1054.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 10:30:00 | 1061.00 | 1062.84 | 1054.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 1057.52 | 1060.25 | 1056.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 15:00:00 | 1057.52 | 1060.25 | 1056.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1063.00 | 1060.33 | 1056.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 10:30:00 | 1066.79 | 1060.65 | 1057.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 12:15:00 | 1069.07 | 1061.44 | 1057.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 11:15:00 | 1129.20 | 1131.56 | 1131.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 11:15:00 | 1129.20 | 1131.56 | 1131.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 13:15:00 | 1126.91 | 1130.53 | 1131.29 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 09:15:00 | 1144.62 | 1132.64 | 1131.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 09:15:00 | 1168.53 | 1139.91 | 1135.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 1165.78 | 1168.62 | 1158.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 14:00:00 | 1165.78 | 1168.62 | 1158.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 12:15:00 | 1162.77 | 1168.06 | 1163.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 12:45:00 | 1162.39 | 1168.06 | 1163.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 1164.47 | 1167.34 | 1163.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 13:30:00 | 1162.60 | 1167.34 | 1163.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 1165.20 | 1166.48 | 1163.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 1167.79 | 1166.48 | 1163.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 1173.57 | 1167.90 | 1164.41 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 10:15:00 | 1162.65 | 1164.26 | 1164.44 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 11:15:00 | 1168.59 | 1165.13 | 1164.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 15:15:00 | 1176.45 | 1169.43 | 1167.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 14:15:00 | 1175.66 | 1178.79 | 1175.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 14:15:00 | 1175.66 | 1178.79 | 1175.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 1175.66 | 1178.79 | 1175.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 15:00:00 | 1175.66 | 1178.79 | 1175.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 1170.00 | 1177.03 | 1174.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 09:30:00 | 1176.80 | 1176.20 | 1174.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 10:30:00 | 1175.93 | 1176.37 | 1175.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 09:15:00 | 1166.06 | 1182.64 | 1184.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 1166.06 | 1182.64 | 1184.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 14:15:00 | 1155.80 | 1169.37 | 1176.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 09:15:00 | 1124.01 | 1119.11 | 1126.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-04 10:00:00 | 1124.01 | 1119.11 | 1126.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 1125.55 | 1120.40 | 1126.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:15:00 | 1127.00 | 1120.40 | 1126.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 11:15:00 | 1126.90 | 1121.70 | 1126.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 12:15:00 | 1129.07 | 1121.70 | 1126.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 1130.70 | 1123.50 | 1126.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 13:00:00 | 1130.70 | 1123.50 | 1126.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 1131.88 | 1126.01 | 1127.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 15:00:00 | 1131.88 | 1126.01 | 1127.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 1123.00 | 1120.32 | 1122.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 11:30:00 | 1118.80 | 1120.05 | 1122.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 09:15:00 | 1128.60 | 1119.67 | 1120.94 | SL hit (close>static) qty=1.00 sl=1126.99 alert=retest2 |

### Cycle 13 — BUY (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 13:15:00 | 1123.26 | 1121.66 | 1121.57 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 10:15:00 | 1117.20 | 1120.97 | 1121.32 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 10:15:00 | 1126.50 | 1121.32 | 1120.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-12 09:15:00 | 1135.54 | 1124.66 | 1122.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 10:15:00 | 1142.75 | 1143.34 | 1137.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-14 11:00:00 | 1142.75 | 1143.34 | 1137.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 1149.28 | 1154.14 | 1149.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 11:45:00 | 1148.84 | 1154.14 | 1149.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 1148.99 | 1153.11 | 1149.75 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 1133.24 | 1146.43 | 1147.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 1125.66 | 1136.49 | 1141.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 13:15:00 | 1132.83 | 1129.23 | 1134.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 13:15:00 | 1132.83 | 1129.23 | 1134.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 1132.83 | 1129.23 | 1134.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:00:00 | 1132.83 | 1129.23 | 1134.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 1135.40 | 1131.00 | 1134.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:15:00 | 1129.82 | 1131.00 | 1134.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 1106.55 | 1126.11 | 1131.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 10:15:00 | 1104.08 | 1126.11 | 1131.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 13:30:00 | 1103.60 | 1114.37 | 1123.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 09:15:00 | 1094.49 | 1111.19 | 1120.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 10:45:00 | 1105.42 | 1100.90 | 1103.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 1105.40 | 1102.58 | 1103.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 12:45:00 | 1105.70 | 1102.58 | 1103.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 13:15:00 | 1098.20 | 1101.70 | 1103.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 09:15:00 | 1095.00 | 1101.27 | 1102.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 11:15:00 | 1097.00 | 1100.03 | 1101.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 10:15:00 | 1121.33 | 1099.93 | 1099.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 1121.33 | 1099.93 | 1099.79 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 12:15:00 | 1094.13 | 1103.84 | 1104.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 15:15:00 | 1091.76 | 1098.33 | 1101.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 09:15:00 | 1098.39 | 1098.34 | 1101.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-04 10:00:00 | 1098.39 | 1098.34 | 1101.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1090.39 | 1088.26 | 1093.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 12:00:00 | 1085.46 | 1087.68 | 1092.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 11:15:00 | 1084.90 | 1086.29 | 1089.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 09:15:00 | 1100.01 | 1087.80 | 1088.37 | SL hit (close>static) qty=1.00 sl=1095.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 10:15:00 | 1102.80 | 1090.80 | 1089.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 1116.99 | 1099.12 | 1095.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 09:15:00 | 1107.36 | 1109.19 | 1103.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 1107.36 | 1109.19 | 1103.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 1107.36 | 1109.19 | 1103.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 09:45:00 | 1106.30 | 1109.19 | 1103.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 14:15:00 | 1105.71 | 1108.05 | 1105.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 14:30:00 | 1105.60 | 1108.05 | 1105.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1100.80 | 1106.20 | 1104.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 11:00:00 | 1102.83 | 1105.52 | 1104.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-13 11:15:00 | 1096.91 | 1103.80 | 1104.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 11:15:00 | 1096.91 | 1103.80 | 1104.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 12:15:00 | 1095.22 | 1102.08 | 1103.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 1098.00 | 1097.70 | 1100.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 1098.00 | 1097.70 | 1100.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 1098.00 | 1097.70 | 1100.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:45:00 | 1100.59 | 1097.70 | 1100.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 11:15:00 | 1096.60 | 1097.63 | 1099.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 11:45:00 | 1097.18 | 1097.63 | 1099.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 1098.80 | 1097.81 | 1099.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 15:00:00 | 1098.80 | 1097.81 | 1099.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 15:15:00 | 1097.00 | 1097.65 | 1099.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:15:00 | 1105.25 | 1097.65 | 1099.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 1109.20 | 1099.96 | 1100.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:45:00 | 1108.54 | 1099.96 | 1100.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 10:15:00 | 1110.00 | 1101.97 | 1101.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 11:15:00 | 1113.00 | 1104.17 | 1102.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 09:15:00 | 1126.12 | 1126.76 | 1118.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 15:15:00 | 1123.57 | 1127.34 | 1122.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 1123.57 | 1127.34 | 1122.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:15:00 | 1116.79 | 1127.34 | 1122.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 1119.80 | 1125.83 | 1122.20 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 13:15:00 | 1112.52 | 1119.48 | 1120.03 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-23 10:15:00 | 1123.88 | 1120.28 | 1120.08 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 1118.51 | 1119.93 | 1119.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 1115.36 | 1119.01 | 1119.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 1118.59 | 1115.98 | 1117.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 1118.59 | 1115.98 | 1117.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 1118.59 | 1115.98 | 1117.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 14:15:00 | 1107.46 | 1114.64 | 1116.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 1052.09 | 1079.46 | 1087.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 1082.24 | 1074.58 | 1080.28 | SL hit (close>ema200) qty=0.50 sl=1074.58 alert=retest2 |

### Cycle 25 — BUY (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 09:15:00 | 1084.59 | 1067.93 | 1066.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 10:15:00 | 1089.79 | 1072.31 | 1068.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 15:15:00 | 1088.62 | 1089.78 | 1083.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 09:15:00 | 1093.58 | 1089.78 | 1083.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 1091.37 | 1093.80 | 1088.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:45:00 | 1089.94 | 1093.80 | 1088.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 1081.99 | 1091.34 | 1088.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-10 09:15:00 | 1081.99 | 1091.34 | 1088.57 | SL hit (close<ema400) qty=1.00 sl=1088.57 alert=retest1 |

### Cycle 26 — SELL (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 13:15:00 | 1082.71 | 1087.21 | 1087.31 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 1093.19 | 1088.32 | 1087.72 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 10:15:00 | 1083.32 | 1087.32 | 1087.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 11:15:00 | 1079.22 | 1085.70 | 1086.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 14:15:00 | 1084.60 | 1084.17 | 1085.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-13 15:00:00 | 1084.60 | 1084.17 | 1085.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 1084.19 | 1084.17 | 1085.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:15:00 | 1087.75 | 1084.17 | 1085.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 1087.58 | 1084.85 | 1085.62 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 09:15:00 | 1091.10 | 1086.24 | 1086.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 10:15:00 | 1093.98 | 1087.78 | 1086.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 14:15:00 | 1119.95 | 1120.34 | 1113.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 15:00:00 | 1119.95 | 1120.34 | 1113.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 1136.11 | 1135.12 | 1130.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-24 09:15:00 | 1146.60 | 1131.14 | 1130.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 09:15:00 | 1140.85 | 1132.84 | 1132.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 12:00:00 | 1138.00 | 1135.09 | 1133.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 14:30:00 | 1139.07 | 1135.13 | 1133.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 1132.80 | 1134.67 | 1133.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 09:15:00 | 1135.54 | 1134.67 | 1133.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 10:00:00 | 1136.55 | 1135.04 | 1134.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 11:15:00 | 1137.77 | 1134.53 | 1133.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 12:15:00 | 1143.40 | 1150.18 | 1151.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 1143.40 | 1150.18 | 1151.01 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 1165.82 | 1151.98 | 1151.39 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 1144.70 | 1151.97 | 1152.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 09:15:00 | 1093.18 | 1140.30 | 1147.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 1124.38 | 1108.93 | 1123.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 1124.38 | 1108.93 | 1123.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 1124.38 | 1108.93 | 1123.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:00:00 | 1124.38 | 1108.93 | 1123.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 1110.40 | 1109.22 | 1122.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 14:45:00 | 1105.69 | 1109.31 | 1118.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-14 11:15:00 | 1120.10 | 1118.30 | 1118.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 1120.10 | 1118.30 | 1118.16 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 14:15:00 | 1114.20 | 1117.74 | 1117.97 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 1122.31 | 1118.34 | 1118.18 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 10:15:00 | 1116.56 | 1117.99 | 1118.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 11:15:00 | 1113.07 | 1117.00 | 1117.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 14:15:00 | 1116.00 | 1115.05 | 1116.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 14:15:00 | 1116.00 | 1115.05 | 1116.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 1116.00 | 1115.05 | 1116.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 1116.00 | 1115.05 | 1116.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 1120.76 | 1116.19 | 1116.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:15:00 | 1132.19 | 1116.19 | 1116.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 09:15:00 | 1132.34 | 1119.42 | 1118.21 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 1115.14 | 1122.72 | 1123.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 09:15:00 | 1106.60 | 1119.49 | 1121.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1125.32 | 1116.62 | 1118.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 1125.32 | 1116.62 | 1118.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1125.32 | 1116.62 | 1118.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:45:00 | 1126.03 | 1116.62 | 1118.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 1126.96 | 1118.69 | 1119.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:30:00 | 1128.43 | 1118.69 | 1119.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 1128.84 | 1120.72 | 1120.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 1133.11 | 1125.11 | 1122.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 13:15:00 | 1128.80 | 1129.36 | 1125.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 13:45:00 | 1127.00 | 1129.36 | 1125.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 1126.40 | 1128.77 | 1125.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 15:00:00 | 1126.40 | 1128.77 | 1125.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 1126.00 | 1128.22 | 1125.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:15:00 | 1126.46 | 1128.22 | 1125.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 1128.77 | 1128.33 | 1126.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:30:00 | 1124.77 | 1128.33 | 1126.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 1177.96 | 1183.74 | 1178.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 10:00:00 | 1177.96 | 1183.74 | 1178.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 1178.49 | 1182.69 | 1178.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:15:00 | 1175.68 | 1182.69 | 1178.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 1173.17 | 1180.79 | 1178.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 12:00:00 | 1173.17 | 1180.79 | 1178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 12:15:00 | 1173.26 | 1179.28 | 1177.59 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-01-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 14:15:00 | 1170.46 | 1176.27 | 1176.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-04 15:15:00 | 1166.49 | 1174.32 | 1175.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 15:15:00 | 1170.47 | 1170.19 | 1172.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-08 09:15:00 | 1172.50 | 1170.19 | 1172.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 1158.07 | 1167.77 | 1171.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 10:15:00 | 1153.11 | 1167.77 | 1171.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 11:15:00 | 1156.18 | 1156.46 | 1161.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 13:30:00 | 1156.41 | 1156.25 | 1160.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 09:15:00 | 1139.21 | 1151.74 | 1154.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 1131.29 | 1138.19 | 1144.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-15 11:15:00 | 1158.30 | 1144.94 | 1143.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 11:15:00 | 1158.30 | 1144.94 | 1143.68 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 09:15:00 | 1136.98 | 1147.64 | 1147.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 15:15:00 | 1127.64 | 1136.65 | 1141.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 13:15:00 | 1130.27 | 1129.19 | 1135.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 14:15:00 | 1130.98 | 1129.19 | 1135.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 1131.59 | 1129.59 | 1133.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 11:45:00 | 1127.47 | 1128.67 | 1132.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 14:00:00 | 1127.88 | 1128.39 | 1131.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 13:30:00 | 1127.80 | 1130.92 | 1131.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 14:00:00 | 1127.59 | 1130.92 | 1131.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 1147.64 | 1133.33 | 1132.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 09:15:00 | 1147.64 | 1133.33 | 1132.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 09:15:00 | 1167.74 | 1142.37 | 1137.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 13:15:00 | 1170.85 | 1173.59 | 1162.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-25 13:30:00 | 1175.37 | 1173.59 | 1162.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1151.76 | 1167.64 | 1162.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:15:00 | 1152.05 | 1167.64 | 1162.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 1159.20 | 1164.94 | 1162.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 13:00:00 | 1162.92 | 1164.54 | 1162.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 13:30:00 | 1161.78 | 1163.79 | 1162.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 14:15:00 | 1165.28 | 1163.79 | 1162.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-13 09:15:00 | 1279.21 | 1260.90 | 1248.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 12:15:00 | 1238.11 | 1250.08 | 1251.49 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 1259.99 | 1251.42 | 1250.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 10:15:00 | 1263.61 | 1253.86 | 1252.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 1274.18 | 1278.14 | 1270.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 10:00:00 | 1274.18 | 1278.14 | 1270.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 1270.82 | 1276.68 | 1270.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:00:00 | 1270.82 | 1276.68 | 1270.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 1274.53 | 1276.25 | 1270.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 12:30:00 | 1277.60 | 1276.55 | 1271.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 13:00:00 | 1277.75 | 1276.55 | 1271.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 11:15:00 | 1267.67 | 1273.15 | 1271.99 | SL hit (close<static) qty=1.00 sl=1268.11 alert=retest2 |

### Cycle 46 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 1264.40 | 1270.43 | 1270.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 1249.32 | 1266.21 | 1269.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 1271.59 | 1263.52 | 1266.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 1271.59 | 1263.52 | 1266.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1271.59 | 1263.52 | 1266.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 1271.59 | 1263.52 | 1266.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 1271.79 | 1265.17 | 1266.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 1279.05 | 1265.17 | 1266.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 1297.20 | 1271.58 | 1269.37 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 1273.90 | 1282.69 | 1282.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 10:15:00 | 1272.80 | 1280.71 | 1282.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 1282.04 | 1280.98 | 1282.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 11:15:00 | 1282.04 | 1280.98 | 1282.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 1282.04 | 1280.98 | 1282.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:00:00 | 1282.04 | 1280.98 | 1282.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 1285.39 | 1281.86 | 1282.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:00:00 | 1285.39 | 1281.86 | 1282.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 1287.98 | 1283.08 | 1282.82 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 09:15:00 | 1276.77 | 1281.80 | 1282.32 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 1289.27 | 1283.56 | 1282.98 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 14:15:00 | 1237.92 | 1275.32 | 1279.39 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 1271.72 | 1261.46 | 1261.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 12:15:00 | 1275.84 | 1268.89 | 1265.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 09:15:00 | 1268.48 | 1272.53 | 1268.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 1268.48 | 1272.53 | 1268.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 1268.48 | 1272.53 | 1268.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:00:00 | 1268.48 | 1272.53 | 1268.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 1266.00 | 1271.23 | 1268.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:00:00 | 1266.00 | 1271.23 | 1268.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 1262.24 | 1269.43 | 1268.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:00:00 | 1262.24 | 1269.43 | 1268.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2024-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 13:15:00 | 1260.17 | 1266.07 | 1266.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 1254.21 | 1262.38 | 1264.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 14:15:00 | 1260.58 | 1260.04 | 1262.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 14:15:00 | 1260.58 | 1260.04 | 1262.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 1260.58 | 1260.04 | 1262.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-13 15:00:00 | 1260.58 | 1260.04 | 1262.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 15:15:00 | 1259.80 | 1259.99 | 1262.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 09:15:00 | 1252.50 | 1259.99 | 1262.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 15:15:00 | 1257.02 | 1256.52 | 1259.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 09:15:00 | 1266.00 | 1257.29 | 1257.51 | SL hit (close>static) qty=1.00 sl=1263.91 alert=retest2 |

### Cycle 55 — BUY (started 2024-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 10:15:00 | 1267.70 | 1259.37 | 1258.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 15:15:00 | 1269.80 | 1264.99 | 1261.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 1248.50 | 1261.69 | 1260.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 1248.50 | 1261.69 | 1260.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 1248.50 | 1261.69 | 1260.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 1248.50 | 1261.69 | 1260.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 1240.76 | 1257.51 | 1258.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 13:15:00 | 1235.30 | 1248.45 | 1253.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 1225.02 | 1223.33 | 1232.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-21 10:00:00 | 1225.02 | 1223.33 | 1232.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 1225.07 | 1224.22 | 1228.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 09:30:00 | 1224.20 | 1224.22 | 1228.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 10:15:00 | 1233.38 | 1226.06 | 1229.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 11:00:00 | 1233.38 | 1226.06 | 1229.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 1240.04 | 1228.85 | 1230.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 11:30:00 | 1241.18 | 1228.85 | 1230.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 1242.81 | 1231.64 | 1231.41 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 15:15:00 | 1226.37 | 1232.67 | 1233.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 10:15:00 | 1221.36 | 1229.18 | 1231.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 1230.08 | 1220.73 | 1225.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 1230.08 | 1220.73 | 1225.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1230.08 | 1220.73 | 1225.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:45:00 | 1234.50 | 1220.73 | 1225.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 1228.04 | 1222.19 | 1225.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:30:00 | 1230.82 | 1222.19 | 1225.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 1227.61 | 1223.28 | 1225.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:45:00 | 1231.01 | 1223.28 | 1225.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 13:15:00 | 1236.00 | 1227.65 | 1227.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 09:15:00 | 1241.05 | 1230.64 | 1228.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 09:15:00 | 1242.51 | 1245.96 | 1242.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 1242.51 | 1245.96 | 1242.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 1242.51 | 1245.96 | 1242.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:00:00 | 1242.51 | 1245.96 | 1242.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 1235.91 | 1243.95 | 1241.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:45:00 | 1234.97 | 1243.95 | 1241.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 1236.02 | 1242.36 | 1241.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:30:00 | 1235.07 | 1242.36 | 1241.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2024-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 13:15:00 | 1235.96 | 1239.91 | 1240.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 14:15:00 | 1225.08 | 1236.94 | 1238.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 13:15:00 | 1230.98 | 1227.62 | 1232.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 13:15:00 | 1230.98 | 1227.62 | 1232.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 1230.98 | 1227.62 | 1232.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:00:00 | 1230.98 | 1227.62 | 1232.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 1227.57 | 1227.61 | 1231.76 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 1242.96 | 1234.66 | 1234.18 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 11:15:00 | 1231.03 | 1235.65 | 1235.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 15:15:00 | 1229.60 | 1232.89 | 1233.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 1235.04 | 1233.32 | 1234.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 1235.04 | 1233.32 | 1234.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1235.04 | 1233.32 | 1234.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:45:00 | 1233.40 | 1233.32 | 1234.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 1236.15 | 1233.89 | 1234.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:45:00 | 1236.82 | 1233.89 | 1234.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 1231.60 | 1233.43 | 1233.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:45:00 | 1230.47 | 1231.94 | 1233.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 1207.98 | 1201.57 | 1201.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 1207.98 | 1201.57 | 1201.08 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 14:15:00 | 1191.50 | 1199.78 | 1200.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 15:15:00 | 1189.00 | 1197.62 | 1199.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 09:15:00 | 1196.82 | 1194.24 | 1196.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 1196.82 | 1194.24 | 1196.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1196.82 | 1194.24 | 1196.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 1196.82 | 1194.24 | 1196.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 1201.25 | 1195.64 | 1196.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:30:00 | 1200.99 | 1195.64 | 1196.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 1201.99 | 1196.91 | 1197.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:15:00 | 1205.28 | 1196.91 | 1197.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 12:15:00 | 1205.00 | 1198.53 | 1197.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 13:15:00 | 1237.08 | 1206.24 | 1201.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 1242.97 | 1245.70 | 1231.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 09:45:00 | 1242.55 | 1245.70 | 1231.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 1246.61 | 1252.94 | 1245.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:45:00 | 1245.00 | 1252.94 | 1245.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 1249.22 | 1252.20 | 1245.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:30:00 | 1246.50 | 1252.20 | 1245.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 1245.88 | 1250.93 | 1245.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:00:00 | 1245.88 | 1250.93 | 1245.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 1242.45 | 1249.24 | 1245.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 1242.45 | 1249.24 | 1245.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 1240.07 | 1247.40 | 1244.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 1249.07 | 1247.40 | 1244.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 11:15:00 | 1249.35 | 1256.86 | 1257.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 1249.35 | 1256.86 | 1257.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 1242.56 | 1254.00 | 1256.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 15:15:00 | 1254.68 | 1252.33 | 1254.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 15:15:00 | 1254.68 | 1252.33 | 1254.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 1254.68 | 1252.33 | 1254.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 1208.74 | 1252.33 | 1254.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 13:15:00 | 1148.30 | 1164.90 | 1173.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 1174.00 | 1166.72 | 1173.63 | SL hit (close>ema200) qty=0.50 sl=1166.72 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 13:15:00 | 1170.55 | 1165.16 | 1164.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 15:15:00 | 1174.58 | 1168.26 | 1166.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 15:15:00 | 1176.60 | 1176.71 | 1172.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 09:15:00 | 1172.44 | 1176.71 | 1172.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1176.09 | 1176.58 | 1172.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1173.80 | 1176.58 | 1172.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1175.43 | 1176.35 | 1172.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:45:00 | 1175.17 | 1176.35 | 1172.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 1176.52 | 1176.34 | 1173.70 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 1160.00 | 1172.05 | 1172.31 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 15:15:00 | 1176.00 | 1171.52 | 1171.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 1194.00 | 1176.02 | 1173.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 10:15:00 | 1188.00 | 1188.17 | 1182.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:45:00 | 1186.01 | 1188.17 | 1182.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1183.20 | 1192.77 | 1188.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:45:00 | 1182.00 | 1192.77 | 1188.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1181.59 | 1190.53 | 1187.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 1181.59 | 1190.53 | 1187.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 13:15:00 | 1176.48 | 1184.77 | 1185.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 1174.74 | 1182.77 | 1184.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 1173.64 | 1172.61 | 1177.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 1173.64 | 1172.61 | 1177.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1152.62 | 1168.61 | 1175.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:45:00 | 1174.31 | 1168.61 | 1175.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1165.99 | 1166.13 | 1173.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 1159.62 | 1164.52 | 1171.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:15:00 | 1160.00 | 1163.71 | 1170.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:15:00 | 1159.92 | 1163.14 | 1169.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 1160.00 | 1161.37 | 1166.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1139.18 | 1156.93 | 1164.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 12:00:00 | 1132.10 | 1151.97 | 1161.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1167.52 | 1153.94 | 1158.33 | SL hit (close>static) qty=1.00 sl=1165.47 alert=retest2 |

### Cycle 71 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 1166.67 | 1161.05 | 1160.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1175.46 | 1164.79 | 1162.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1209.09 | 1215.08 | 1203.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:45:00 | 1207.28 | 1215.08 | 1203.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 1208.36 | 1210.91 | 1205.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 1208.36 | 1210.91 | 1205.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1211.33 | 1210.81 | 1207.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:30:00 | 1203.93 | 1210.81 | 1207.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1210.79 | 1211.03 | 1207.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:45:00 | 1209.00 | 1211.03 | 1207.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1211.21 | 1211.07 | 1208.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:45:00 | 1208.38 | 1211.07 | 1208.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1211.11 | 1211.18 | 1208.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 1211.60 | 1211.18 | 1208.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1212.44 | 1214.10 | 1211.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 1212.00 | 1214.10 | 1211.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1212.02 | 1213.68 | 1211.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:30:00 | 1208.98 | 1213.68 | 1211.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 1212.55 | 1213.45 | 1211.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:45:00 | 1212.02 | 1213.45 | 1211.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 1212.36 | 1213.24 | 1211.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 12:45:00 | 1211.01 | 1213.24 | 1211.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 1215.57 | 1213.70 | 1212.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 14:30:00 | 1216.12 | 1214.46 | 1212.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 09:15:00 | 1201.60 | 1212.29 | 1212.04 | SL hit (close<static) qty=1.00 sl=1212.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 1202.35 | 1210.30 | 1211.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 13:15:00 | 1198.81 | 1205.39 | 1208.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 10:15:00 | 1192.00 | 1191.85 | 1197.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 10:15:00 | 1192.00 | 1191.85 | 1197.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1192.00 | 1191.85 | 1197.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 1192.24 | 1191.85 | 1197.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 1192.01 | 1191.53 | 1195.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:30:00 | 1193.25 | 1191.53 | 1195.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1194.30 | 1192.09 | 1195.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:45:00 | 1196.25 | 1192.09 | 1195.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1196.19 | 1192.91 | 1195.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 1199.44 | 1192.91 | 1195.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1208.00 | 1195.92 | 1196.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 1208.00 | 1195.92 | 1196.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 1206.00 | 1197.94 | 1197.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 15:15:00 | 1218.00 | 1205.82 | 1202.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 12:15:00 | 1207.35 | 1207.53 | 1204.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 1207.35 | 1207.53 | 1204.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 1207.35 | 1207.53 | 1204.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:00:00 | 1207.35 | 1207.53 | 1204.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 1208.06 | 1207.64 | 1204.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:30:00 | 1206.00 | 1207.64 | 1204.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1206.00 | 1209.20 | 1206.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 1206.00 | 1209.20 | 1206.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1213.59 | 1210.08 | 1207.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 11:15:00 | 1215.96 | 1210.08 | 1207.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 12:15:00 | 1215.99 | 1210.66 | 1207.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 15:00:00 | 1215.63 | 1213.01 | 1209.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:15:00 | 1240.05 | 1213.21 | 1209.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1239.24 | 1218.41 | 1212.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 15:00:00 | 1248.12 | 1235.59 | 1224.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 1272.00 | 1237.87 | 1226.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-12 13:15:00 | 1337.56 | 1322.88 | 1316.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 1327.60 | 1336.40 | 1337.45 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 1352.80 | 1338.17 | 1336.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 1356.80 | 1348.94 | 1343.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 10:15:00 | 1348.71 | 1348.90 | 1343.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 10:15:00 | 1348.71 | 1348.90 | 1343.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 1348.71 | 1348.90 | 1343.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:45:00 | 1344.48 | 1348.90 | 1343.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1353.17 | 1363.24 | 1359.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 10:30:00 | 1361.79 | 1363.92 | 1360.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 14:30:00 | 1364.00 | 1369.16 | 1366.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:30:00 | 1363.39 | 1371.98 | 1368.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 12:15:00 | 1359.91 | 1366.19 | 1367.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 1359.91 | 1366.19 | 1367.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 11:15:00 | 1355.40 | 1361.06 | 1363.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 1370.68 | 1359.47 | 1361.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 1370.68 | 1359.47 | 1361.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1370.68 | 1359.47 | 1361.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 1370.68 | 1359.47 | 1361.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1374.15 | 1362.40 | 1362.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:30:00 | 1373.61 | 1362.40 | 1362.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 11:15:00 | 1371.78 | 1364.28 | 1363.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 12:15:00 | 1377.03 | 1366.83 | 1364.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1380.60 | 1384.08 | 1378.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 10:15:00 | 1380.60 | 1384.08 | 1378.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 1380.60 | 1384.08 | 1378.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 1380.60 | 1384.08 | 1378.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 1380.60 | 1383.38 | 1378.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:30:00 | 1376.58 | 1383.38 | 1378.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 12:15:00 | 1374.65 | 1381.64 | 1378.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 12:45:00 | 1371.73 | 1381.64 | 1378.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 13:15:00 | 1373.12 | 1379.93 | 1377.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 14:00:00 | 1373.12 | 1379.93 | 1377.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 1361.60 | 1376.27 | 1376.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 15:00:00 | 1361.60 | 1376.27 | 1376.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 15:15:00 | 1363.98 | 1373.81 | 1375.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 1360.02 | 1370.01 | 1372.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 1378.91 | 1370.19 | 1372.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 1378.91 | 1370.19 | 1372.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1378.91 | 1370.19 | 1372.40 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 1382.49 | 1375.01 | 1374.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1401.19 | 1384.25 | 1379.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 1388.18 | 1389.29 | 1384.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 1388.18 | 1389.29 | 1384.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1387.00 | 1388.83 | 1384.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1400.40 | 1388.83 | 1384.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 1382.48 | 1394.08 | 1391.78 | SL hit (close<static) qty=1.00 sl=1382.89 alert=retest2 |

### Cycle 80 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 1376.92 | 1388.66 | 1389.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 12:15:00 | 1367.70 | 1381.03 | 1384.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 1373.26 | 1363.78 | 1369.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 1373.26 | 1363.78 | 1369.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1373.26 | 1363.78 | 1369.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 1373.98 | 1363.78 | 1369.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1377.02 | 1366.43 | 1369.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:45:00 | 1378.61 | 1366.43 | 1369.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 13:15:00 | 1377.49 | 1373.00 | 1372.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 1381.05 | 1374.61 | 1373.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 1389.55 | 1403.29 | 1395.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 1389.55 | 1403.29 | 1395.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1389.55 | 1403.29 | 1395.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:00:00 | 1389.55 | 1403.29 | 1395.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1387.92 | 1400.21 | 1394.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 1387.92 | 1400.21 | 1394.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1395.31 | 1397.38 | 1394.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:30:00 | 1393.00 | 1397.38 | 1394.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1394.00 | 1396.70 | 1394.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:45:00 | 1393.66 | 1396.70 | 1394.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1395.98 | 1396.56 | 1394.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1396.80 | 1396.56 | 1394.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1387.78 | 1394.80 | 1394.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 1387.78 | 1394.80 | 1394.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1392.54 | 1394.35 | 1394.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:15:00 | 1394.00 | 1394.35 | 1394.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 12:15:00 | 1391.89 | 1393.77 | 1393.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 1391.89 | 1393.77 | 1393.84 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 13:15:00 | 1395.37 | 1394.09 | 1393.98 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 1391.56 | 1393.58 | 1393.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 15:15:00 | 1388.20 | 1392.51 | 1393.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1394.34 | 1390.25 | 1391.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1394.34 | 1390.25 | 1391.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1394.34 | 1390.25 | 1391.17 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 13:15:00 | 1397.83 | 1392.78 | 1392.20 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 1386.81 | 1391.60 | 1391.79 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 1398.21 | 1391.79 | 1391.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 14:15:00 | 1399.47 | 1393.32 | 1392.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 09:15:00 | 1384.42 | 1392.57 | 1392.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 1384.42 | 1392.57 | 1392.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1384.42 | 1392.57 | 1392.27 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1386.22 | 1391.30 | 1391.72 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1406.40 | 1391.48 | 1390.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 11:15:00 | 1408.00 | 1396.64 | 1393.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 15:15:00 | 1398.88 | 1400.31 | 1396.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 09:15:00 | 1396.89 | 1400.31 | 1396.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 1390.92 | 1398.43 | 1396.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:00:00 | 1390.92 | 1398.43 | 1396.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1383.61 | 1395.47 | 1394.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 1383.61 | 1395.47 | 1394.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 11:15:00 | 1379.04 | 1392.18 | 1393.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 12:15:00 | 1373.00 | 1388.35 | 1391.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 1367.76 | 1362.57 | 1369.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 10:00:00 | 1367.76 | 1362.57 | 1369.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1357.00 | 1361.46 | 1368.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:15:00 | 1354.80 | 1361.46 | 1368.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:15:00 | 1356.33 | 1360.69 | 1367.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 14:15:00 | 1332.09 | 1331.41 | 1331.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 14:15:00 | 1332.09 | 1331.41 | 1331.41 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 1330.00 | 1331.38 | 1331.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 12:15:00 | 1328.78 | 1330.86 | 1331.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 13:15:00 | 1331.16 | 1330.92 | 1331.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 13:15:00 | 1331.16 | 1330.92 | 1331.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 1331.16 | 1330.92 | 1331.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:45:00 | 1331.70 | 1330.92 | 1331.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1329.09 | 1330.55 | 1331.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 10:15:00 | 1325.92 | 1329.76 | 1330.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 11:00:00 | 1326.08 | 1329.03 | 1330.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 11:30:00 | 1324.79 | 1328.10 | 1329.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 13:15:00 | 1331.69 | 1327.71 | 1329.12 | SL hit (close>static) qty=1.00 sl=1331.16 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1322.46 | 1314.30 | 1313.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 11:15:00 | 1328.40 | 1317.12 | 1314.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 1321.54 | 1323.70 | 1319.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:00:00 | 1321.54 | 1323.70 | 1319.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1333.62 | 1325.69 | 1320.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:30:00 | 1325.45 | 1325.69 | 1320.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1326.48 | 1328.09 | 1324.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 1339.12 | 1327.28 | 1324.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 1338.51 | 1347.17 | 1347.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 1338.51 | 1347.17 | 1347.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 1327.62 | 1337.60 | 1342.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 09:15:00 | 1336.05 | 1335.82 | 1340.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 1336.05 | 1335.82 | 1340.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1336.05 | 1335.82 | 1340.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:30:00 | 1339.12 | 1335.82 | 1340.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 1332.76 | 1335.21 | 1339.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 12:45:00 | 1328.00 | 1332.31 | 1337.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 14:15:00 | 1328.52 | 1326.67 | 1331.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 1340.08 | 1332.76 | 1332.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1340.08 | 1332.76 | 1332.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 12:15:00 | 1342.03 | 1334.61 | 1333.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 1332.23 | 1335.16 | 1333.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 1332.23 | 1335.16 | 1333.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 1332.23 | 1335.16 | 1333.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 1332.23 | 1335.16 | 1333.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 1332.00 | 1334.53 | 1333.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:15:00 | 1327.64 | 1334.53 | 1333.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 09:15:00 | 1325.73 | 1332.77 | 1333.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 1320.26 | 1328.77 | 1331.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 13:15:00 | 1321.49 | 1319.71 | 1323.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 13:15:00 | 1321.49 | 1319.71 | 1323.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 1321.49 | 1319.71 | 1323.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:00:00 | 1321.49 | 1319.71 | 1323.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 1319.77 | 1319.72 | 1323.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:45:00 | 1321.23 | 1319.72 | 1323.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1326.26 | 1321.07 | 1323.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 1324.77 | 1321.07 | 1323.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1328.00 | 1322.46 | 1323.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:45:00 | 1327.20 | 1322.46 | 1323.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 1331.31 | 1325.56 | 1324.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 13:15:00 | 1335.18 | 1327.49 | 1325.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 1326.20 | 1328.75 | 1327.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1326.20 | 1328.75 | 1327.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1326.20 | 1328.75 | 1327.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 1326.20 | 1328.75 | 1327.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1326.44 | 1328.29 | 1326.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 1325.00 | 1328.29 | 1326.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1325.46 | 1327.72 | 1326.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 1325.46 | 1327.72 | 1326.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 1324.78 | 1327.13 | 1326.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:30:00 | 1324.00 | 1327.13 | 1326.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 1326.06 | 1326.92 | 1326.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:30:00 | 1326.00 | 1326.92 | 1326.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 14:15:00 | 1321.49 | 1325.83 | 1326.11 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 09:15:00 | 1335.80 | 1327.53 | 1326.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 10:15:00 | 1336.89 | 1329.40 | 1327.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 1334.40 | 1338.91 | 1334.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 10:15:00 | 1334.40 | 1338.91 | 1334.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1334.40 | 1338.91 | 1334.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 1334.40 | 1338.91 | 1334.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1339.99 | 1339.12 | 1334.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:15:00 | 1341.00 | 1339.12 | 1334.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 15:00:00 | 1341.65 | 1338.34 | 1335.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 09:45:00 | 1345.20 | 1341.54 | 1337.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 10:30:00 | 1341.00 | 1345.08 | 1342.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 1341.46 | 1344.36 | 1342.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 1341.46 | 1344.36 | 1342.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 1342.91 | 1344.07 | 1342.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:45:00 | 1341.80 | 1344.07 | 1342.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 1345.85 | 1344.43 | 1342.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:45:00 | 1345.46 | 1344.43 | 1342.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 1339.42 | 1343.42 | 1342.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:00:00 | 1339.42 | 1343.42 | 1342.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1338.81 | 1342.50 | 1342.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 1339.00 | 1342.50 | 1342.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 1342.20 | 1342.44 | 1342.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:30:00 | 1340.00 | 1342.44 | 1342.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-22 10:15:00 | 1338.45 | 1341.64 | 1341.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 1338.45 | 1341.64 | 1341.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 1331.00 | 1339.13 | 1340.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 1314.60 | 1307.14 | 1313.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 10:15:00 | 1314.60 | 1307.14 | 1313.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1314.60 | 1307.14 | 1313.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:45:00 | 1315.05 | 1307.14 | 1313.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1315.65 | 1308.84 | 1313.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:30:00 | 1316.30 | 1308.84 | 1313.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 1312.10 | 1309.49 | 1313.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:30:00 | 1318.15 | 1309.49 | 1313.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 1312.40 | 1310.07 | 1313.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:30:00 | 1300.75 | 1306.66 | 1311.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:15:00 | 1235.71 | 1274.35 | 1289.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-31 11:15:00 | 1260.70 | 1256.42 | 1268.32 | SL hit (close>ema200) qty=0.50 sl=1256.42 alert=retest2 |

### Cycle 101 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 1276.90 | 1266.58 | 1266.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1301.05 | 1273.47 | 1269.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1287.10 | 1294.05 | 1285.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1287.10 | 1294.05 | 1285.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1287.10 | 1294.05 | 1285.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1287.10 | 1294.05 | 1285.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1286.20 | 1292.48 | 1285.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 1283.40 | 1292.48 | 1285.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1291.05 | 1292.19 | 1285.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:30:00 | 1291.05 | 1292.19 | 1285.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1288.45 | 1290.58 | 1286.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 1288.45 | 1290.58 | 1286.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1288.00 | 1290.06 | 1286.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1276.35 | 1290.06 | 1286.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1271.75 | 1286.40 | 1285.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 1270.80 | 1286.40 | 1285.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1273.00 | 1283.72 | 1284.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1260.10 | 1279.00 | 1281.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 13:15:00 | 1282.50 | 1279.46 | 1281.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 13:15:00 | 1282.50 | 1279.46 | 1281.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 1282.50 | 1279.46 | 1281.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 1282.50 | 1279.46 | 1281.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 1284.35 | 1280.44 | 1281.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 1284.35 | 1280.44 | 1281.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 1280.00 | 1280.35 | 1281.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 1279.80 | 1280.35 | 1281.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1284.20 | 1281.12 | 1281.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:45:00 | 1291.00 | 1281.12 | 1281.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 1293.15 | 1283.53 | 1282.96 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 1278.30 | 1283.00 | 1283.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 1275.50 | 1281.50 | 1282.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 1214.30 | 1206.53 | 1221.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 10:00:00 | 1214.30 | 1206.53 | 1221.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 1221.40 | 1210.78 | 1221.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:45:00 | 1220.80 | 1210.78 | 1221.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 1225.00 | 1213.62 | 1221.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:00:00 | 1225.00 | 1213.62 | 1221.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 1226.50 | 1216.20 | 1221.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:45:00 | 1228.60 | 1216.20 | 1221.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1208.35 | 1201.85 | 1207.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 1208.35 | 1201.85 | 1207.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1214.05 | 1204.29 | 1208.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 1214.05 | 1204.29 | 1208.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1218.35 | 1207.10 | 1208.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 1218.35 | 1207.10 | 1208.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1217.00 | 1211.09 | 1210.50 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 1209.85 | 1211.70 | 1211.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 1208.15 | 1210.99 | 1211.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 15:15:00 | 1212.00 | 1211.19 | 1211.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 15:15:00 | 1212.00 | 1211.19 | 1211.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1212.00 | 1211.19 | 1211.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 1205.75 | 1211.19 | 1211.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1204.35 | 1209.82 | 1210.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 11:00:00 | 1199.05 | 1207.67 | 1209.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 13:45:00 | 1200.50 | 1204.47 | 1207.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 15:00:00 | 1200.05 | 1203.58 | 1206.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 09:30:00 | 1198.15 | 1202.30 | 1205.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1203.25 | 1197.13 | 1200.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 1221.00 | 1205.44 | 1203.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 1221.00 | 1205.44 | 1203.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 1221.70 | 1215.05 | 1209.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 09:15:00 | 1213.80 | 1220.77 | 1216.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 1213.80 | 1220.77 | 1216.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1213.80 | 1220.77 | 1216.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 1213.80 | 1220.77 | 1216.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1215.00 | 1219.62 | 1216.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 1213.20 | 1219.62 | 1216.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1214.50 | 1218.60 | 1216.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1212.60 | 1218.60 | 1216.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1216.10 | 1218.10 | 1216.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:30:00 | 1218.60 | 1218.58 | 1216.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 15:15:00 | 1216.50 | 1217.81 | 1216.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 10:00:00 | 1219.45 | 1217.93 | 1216.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 14:15:00 | 1240.10 | 1243.01 | 1243.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 14:15:00 | 1240.10 | 1243.01 | 1243.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 1232.65 | 1237.88 | 1239.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 12:15:00 | 1239.90 | 1236.58 | 1238.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 12:15:00 | 1239.90 | 1236.58 | 1238.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1239.90 | 1236.58 | 1238.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:00:00 | 1239.90 | 1236.58 | 1238.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 1239.60 | 1237.18 | 1238.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:00:00 | 1239.60 | 1237.18 | 1238.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 1247.00 | 1239.15 | 1239.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 1247.00 | 1239.15 | 1239.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 1240.40 | 1239.40 | 1239.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 1237.00 | 1239.40 | 1239.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 12:15:00 | 1244.90 | 1240.52 | 1239.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 12:15:00 | 1244.90 | 1240.52 | 1239.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 13:15:00 | 1247.80 | 1241.98 | 1240.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 11:15:00 | 1257.90 | 1260.65 | 1254.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 11:45:00 | 1257.65 | 1260.65 | 1254.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 1254.05 | 1259.33 | 1254.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 1254.05 | 1259.33 | 1254.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1246.70 | 1256.80 | 1253.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:45:00 | 1246.45 | 1256.80 | 1253.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 1245.50 | 1254.54 | 1252.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:15:00 | 1251.80 | 1254.54 | 1252.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1356.15 | 1352.39 | 1343.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 1354.30 | 1352.39 | 1343.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 1347.60 | 1351.51 | 1345.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:45:00 | 1345.60 | 1351.51 | 1345.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 1348.45 | 1350.90 | 1345.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 1349.65 | 1350.90 | 1345.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1375.55 | 1357.52 | 1350.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 10:45:00 | 1384.60 | 1362.75 | 1353.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 12:30:00 | 1388.30 | 1380.92 | 1371.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 14:00:00 | 1383.65 | 1381.46 | 1372.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 14:45:00 | 1385.30 | 1380.15 | 1372.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1379.00 | 1379.92 | 1373.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 1375.00 | 1379.92 | 1373.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1369.70 | 1377.88 | 1373.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:45:00 | 1385.70 | 1379.30 | 1374.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 1363.45 | 1376.60 | 1376.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 11:15:00 | 1363.45 | 1376.60 | 1376.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 10:15:00 | 1358.75 | 1369.09 | 1371.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 15:15:00 | 1352.75 | 1348.11 | 1354.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 15:15:00 | 1352.75 | 1348.11 | 1354.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 1352.75 | 1348.11 | 1354.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 1350.00 | 1348.11 | 1354.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1360.55 | 1350.60 | 1355.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:15:00 | 1365.05 | 1350.60 | 1355.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1367.60 | 1354.00 | 1356.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:15:00 | 1370.05 | 1354.00 | 1356.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 1368.45 | 1358.26 | 1358.16 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 1351.35 | 1357.89 | 1358.07 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 09:15:00 | 1377.85 | 1361.06 | 1359.43 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 1357.05 | 1366.76 | 1367.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 1355.00 | 1364.41 | 1366.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 12:15:00 | 1329.50 | 1329.06 | 1337.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 12:45:00 | 1330.00 | 1329.06 | 1337.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 1339.45 | 1331.14 | 1337.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:00:00 | 1339.45 | 1331.14 | 1337.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 1338.75 | 1332.66 | 1337.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 15:00:00 | 1338.75 | 1332.66 | 1337.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 1333.50 | 1332.83 | 1337.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 1326.90 | 1332.83 | 1337.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1327.00 | 1331.66 | 1336.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:15:00 | 1324.50 | 1331.66 | 1336.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:15:00 | 1258.27 | 1278.96 | 1288.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 14:15:00 | 1192.05 | 1214.09 | 1237.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 1217.20 | 1200.85 | 1199.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 11:15:00 | 1227.55 | 1213.24 | 1209.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 1231.55 | 1235.73 | 1230.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 1231.55 | 1235.73 | 1230.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1231.55 | 1235.73 | 1230.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 1226.85 | 1235.73 | 1230.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1237.80 | 1236.14 | 1230.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 1239.80 | 1236.16 | 1231.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:30:00 | 1240.90 | 1236.02 | 1232.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 15:00:00 | 1240.55 | 1236.93 | 1232.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 1217.20 | 1233.68 | 1232.08 | SL hit (close<static) qty=1.00 sl=1229.85 alert=retest2 |

### Cycle 116 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1225.20 | 1229.94 | 1230.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1214.65 | 1223.21 | 1226.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 1214.90 | 1213.61 | 1219.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 09:15:00 | 1209.95 | 1213.61 | 1219.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1215.00 | 1213.32 | 1217.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:30:00 | 1215.50 | 1213.32 | 1217.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1215.70 | 1213.80 | 1217.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 14:00:00 | 1215.70 | 1213.80 | 1217.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 1216.20 | 1214.28 | 1217.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 14:45:00 | 1215.15 | 1214.28 | 1217.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 1216.85 | 1214.79 | 1217.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 1237.20 | 1214.79 | 1217.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1231.60 | 1218.15 | 1218.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:30:00 | 1224.90 | 1218.15 | 1218.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 1223.35 | 1219.19 | 1218.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 12:15:00 | 1235.00 | 1223.71 | 1221.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 14:15:00 | 1223.45 | 1224.84 | 1222.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 14:15:00 | 1223.45 | 1224.84 | 1222.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 1223.45 | 1224.84 | 1222.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 15:00:00 | 1223.45 | 1224.84 | 1222.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 15:15:00 | 1222.80 | 1224.43 | 1222.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:15:00 | 1210.50 | 1224.43 | 1222.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1210.70 | 1221.68 | 1221.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 1210.70 | 1221.68 | 1221.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 1201.95 | 1217.74 | 1219.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 1188.70 | 1208.40 | 1214.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 1200.45 | 1198.69 | 1204.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 1200.45 | 1198.69 | 1204.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1189.70 | 1197.00 | 1202.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 1185.35 | 1197.00 | 1202.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 1149.20 | 1196.19 | 1199.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 14:15:00 | 1126.08 | 1142.88 | 1152.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 1115.80 | 1114.82 | 1122.56 | SL hit (close>ema200) qty=0.50 sl=1114.82 alert=retest2 |

### Cycle 119 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 1126.50 | 1121.30 | 1121.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 11:15:00 | 1133.25 | 1126.42 | 1124.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1135.60 | 1135.82 | 1131.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 1135.60 | 1135.82 | 1131.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1133.20 | 1134.87 | 1132.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 1123.70 | 1134.87 | 1132.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1130.00 | 1133.90 | 1131.98 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 1126.40 | 1130.91 | 1131.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 1120.75 | 1128.88 | 1130.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 1107.25 | 1106.36 | 1113.02 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1103.00 | 1106.36 | 1113.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1109.05 | 1105.46 | 1109.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 1109.05 | 1105.46 | 1109.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1107.00 | 1105.77 | 1109.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 1135.30 | 1105.77 | 1109.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1144.30 | 1113.48 | 1112.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 121 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1144.30 | 1113.48 | 1112.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 14:15:00 | 1150.70 | 1135.36 | 1124.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1175.35 | 1196.92 | 1191.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 1175.35 | 1196.92 | 1191.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1175.35 | 1196.92 | 1191.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1175.35 | 1196.92 | 1191.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1175.75 | 1192.68 | 1189.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 1169.00 | 1192.68 | 1189.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 1173.65 | 1185.74 | 1186.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 1156.45 | 1176.23 | 1181.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 1162.80 | 1158.68 | 1165.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 1162.80 | 1158.68 | 1165.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1162.80 | 1158.68 | 1165.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1162.80 | 1158.68 | 1165.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1156.00 | 1150.08 | 1155.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 1146.50 | 1150.08 | 1155.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1155.05 | 1151.07 | 1155.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:45:00 | 1152.65 | 1152.09 | 1155.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 1170.00 | 1151.61 | 1151.62 | SL hit (close>static) qty=1.00 sl=1164.40 alert=retest2 |

### Cycle 123 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 1170.20 | 1155.33 | 1153.31 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1086.35 | 1141.44 | 1148.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1074.35 | 1108.07 | 1125.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1105.90 | 1096.49 | 1107.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 10:15:00 | 1105.90 | 1096.49 | 1107.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 1105.90 | 1096.49 | 1107.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 1105.90 | 1096.49 | 1107.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1112.05 | 1101.37 | 1108.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:00:00 | 1112.05 | 1101.37 | 1108.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1109.40 | 1102.97 | 1108.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 1110.40 | 1102.97 | 1108.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1108.05 | 1103.99 | 1108.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:00:00 | 1108.05 | 1103.99 | 1108.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 1109.85 | 1105.16 | 1108.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1079.80 | 1105.16 | 1108.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1125.70 | 1102.97 | 1104.22 | SL hit (close>static) qty=1.00 sl=1112.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 1119.85 | 1106.34 | 1105.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1140.90 | 1118.50 | 1112.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 1150.90 | 1151.10 | 1140.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 1159.70 | 1151.10 | 1140.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 1172.60 | 1171.19 | 1162.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 1179.70 | 1174.09 | 1168.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 1174.00 | 1181.20 | 1181.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 1174.00 | 1181.20 | 1181.88 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1202.20 | 1182.13 | 1181.69 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 1172.00 | 1185.82 | 1186.01 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 1191.00 | 1184.36 | 1184.21 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 1176.30 | 1182.75 | 1183.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 1175.10 | 1180.76 | 1182.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 13:15:00 | 1187.40 | 1182.09 | 1182.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 13:15:00 | 1187.40 | 1182.09 | 1182.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 1187.40 | 1182.09 | 1182.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:00:00 | 1187.40 | 1182.09 | 1182.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 1184.00 | 1182.47 | 1182.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 09:30:00 | 1174.70 | 1181.68 | 1182.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:15:00 | 1172.90 | 1181.68 | 1182.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1155.30 | 1173.95 | 1177.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 1183.20 | 1159.05 | 1157.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1183.20 | 1159.05 | 1157.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1188.50 | 1168.85 | 1162.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1205.40 | 1213.70 | 1204.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1205.40 | 1213.70 | 1204.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1205.40 | 1213.70 | 1204.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 1206.50 | 1213.70 | 1204.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1209.10 | 1212.78 | 1205.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1226.00 | 1215.12 | 1207.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 1217.30 | 1225.81 | 1221.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 1222.30 | 1221.20 | 1220.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1214.10 | 1224.24 | 1224.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 1214.10 | 1224.24 | 1224.95 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1231.20 | 1224.93 | 1224.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1240.40 | 1230.60 | 1228.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 15:15:00 | 1242.00 | 1242.56 | 1238.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1247.40 | 1242.56 | 1238.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:30:00 | 1246.40 | 1243.68 | 1239.92 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 1242.50 | 1243.44 | 1240.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 1242.50 | 1243.44 | 1240.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1238.70 | 1242.49 | 1240.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 1238.70 | 1242.49 | 1240.02 | SL hit (close<ema400) qty=1.00 sl=1240.02 alert=retest1 |

### Cycle 134 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 1245.70 | 1245.82 | 1245.83 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 1249.10 | 1246.48 | 1246.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1256.50 | 1249.14 | 1247.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 1251.70 | 1252.64 | 1250.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 14:15:00 | 1251.70 | 1252.64 | 1250.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1251.70 | 1252.64 | 1250.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 1251.50 | 1252.64 | 1250.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1251.00 | 1252.31 | 1250.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1288.60 | 1252.31 | 1250.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 1347.00 | 1351.62 | 1351.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 14:15:00 | 1347.00 | 1351.62 | 1351.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 1334.70 | 1347.98 | 1350.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1318.60 | 1318.44 | 1327.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 1318.60 | 1318.44 | 1327.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1329.90 | 1321.30 | 1327.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1329.90 | 1321.30 | 1327.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1326.90 | 1322.42 | 1327.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 1322.80 | 1324.11 | 1327.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 1331.90 | 1325.67 | 1327.72 | SL hit (close>static) qty=1.00 sl=1331.60 alert=retest2 |

### Cycle 137 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1338.30 | 1326.08 | 1324.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1344.30 | 1329.73 | 1326.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1327.50 | 1337.10 | 1334.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 1327.50 | 1337.10 | 1334.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1327.50 | 1337.10 | 1334.82 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1316.60 | 1330.04 | 1331.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 09:15:00 | 1312.50 | 1322.25 | 1326.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 15:15:00 | 1282.00 | 1281.25 | 1289.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:15:00 | 1275.50 | 1281.25 | 1289.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1281.20 | 1275.51 | 1280.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1281.20 | 1275.51 | 1280.41 | SL hit (close>ema400) qty=1.00 sl=1280.41 alert=retest1 |

### Cycle 139 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 1288.70 | 1282.80 | 1282.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 15:15:00 | 1295.00 | 1285.24 | 1283.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1287.80 | 1301.87 | 1297.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1287.80 | 1301.87 | 1297.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1287.80 | 1301.87 | 1297.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:15:00 | 1285.50 | 1301.87 | 1297.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1277.30 | 1296.96 | 1296.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 1277.30 | 1296.96 | 1296.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 1279.50 | 1293.47 | 1294.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 1271.40 | 1279.49 | 1285.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 1269.00 | 1268.14 | 1275.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:45:00 | 1269.80 | 1268.14 | 1275.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1256.30 | 1252.27 | 1257.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 1256.50 | 1252.27 | 1257.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1256.50 | 1253.12 | 1257.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 1256.40 | 1253.12 | 1257.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1260.50 | 1255.14 | 1257.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1260.50 | 1255.14 | 1257.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1257.90 | 1255.69 | 1257.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1256.40 | 1255.69 | 1257.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 13:45:00 | 1257.20 | 1256.46 | 1257.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 15:15:00 | 1259.50 | 1257.57 | 1257.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 1259.50 | 1257.57 | 1257.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 1260.00 | 1258.06 | 1257.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 1261.30 | 1263.26 | 1261.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 1261.30 | 1263.26 | 1261.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1261.30 | 1263.26 | 1261.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 1259.90 | 1263.26 | 1261.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1251.90 | 1260.99 | 1260.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 1251.90 | 1260.99 | 1260.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1257.30 | 1260.25 | 1260.10 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 1257.10 | 1259.62 | 1259.83 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 1261.20 | 1260.05 | 1259.99 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 1258.30 | 1259.70 | 1259.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 1252.80 | 1258.32 | 1259.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1257.10 | 1257.02 | 1258.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 1257.10 | 1257.02 | 1258.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1257.70 | 1256.99 | 1258.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 1257.50 | 1256.99 | 1258.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1259.50 | 1257.49 | 1258.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1259.50 | 1257.49 | 1258.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1256.10 | 1257.21 | 1258.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1251.60 | 1257.21 | 1258.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1247.50 | 1255.27 | 1257.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 1244.90 | 1253.36 | 1256.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 1242.50 | 1253.36 | 1256.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:45:00 | 1242.70 | 1247.68 | 1252.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1276.90 | 1249.86 | 1248.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1276.90 | 1249.86 | 1248.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 1288.60 | 1282.52 | 1276.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 1278.70 | 1289.31 | 1283.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 1278.70 | 1289.31 | 1283.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1278.70 | 1289.31 | 1283.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 1278.70 | 1289.31 | 1283.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1287.90 | 1289.03 | 1284.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 1292.00 | 1289.03 | 1284.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:45:00 | 1289.50 | 1288.61 | 1285.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1266.50 | 1284.73 | 1284.21 | SL hit (close<static) qty=1.00 sl=1278.30 alert=retest2 |

### Cycle 146 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1277.40 | 1283.26 | 1283.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 1243.40 | 1270.12 | 1276.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1202.00 | 1194.62 | 1203.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 1202.00 | 1194.62 | 1203.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1198.00 | 1195.29 | 1202.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 1197.60 | 1196.99 | 1202.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 1210.70 | 1203.80 | 1204.15 | SL hit (close>static) qty=1.00 sl=1210.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 15:15:00 | 1215.00 | 1206.04 | 1205.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 1224.60 | 1209.75 | 1206.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 1212.40 | 1217.25 | 1213.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1212.40 | 1217.25 | 1213.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1212.40 | 1217.25 | 1213.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:30:00 | 1215.80 | 1216.52 | 1213.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 11:15:00 | 1215.60 | 1216.52 | 1213.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1241.40 | 1248.61 | 1249.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 1241.40 | 1248.61 | 1249.01 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 1261.00 | 1250.06 | 1249.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 1269.70 | 1253.99 | 1251.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 15:15:00 | 1275.00 | 1275.70 | 1268.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 1284.10 | 1277.66 | 1269.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1264.20 | 1278.89 | 1275.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1264.20 | 1278.89 | 1275.20 | SL hit (close<ema400) qty=1.00 sl=1275.20 alert=retest1 |

### Cycle 150 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 1264.90 | 1272.18 | 1272.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 1261.10 | 1267.34 | 1270.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 14:15:00 | 1260.10 | 1259.42 | 1263.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 15:00:00 | 1260.10 | 1259.42 | 1263.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1247.50 | 1257.74 | 1262.47 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1274.40 | 1263.63 | 1262.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1280.00 | 1268.08 | 1265.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1269.90 | 1272.87 | 1268.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1269.90 | 1272.87 | 1268.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1269.90 | 1272.87 | 1268.80 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 1256.70 | 1266.62 | 1267.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 15:15:00 | 1250.00 | 1263.30 | 1265.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 11:15:00 | 1262.10 | 1261.63 | 1264.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 11:15:00 | 1262.10 | 1261.63 | 1264.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1262.10 | 1261.63 | 1264.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:30:00 | 1264.70 | 1261.63 | 1264.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1264.20 | 1262.14 | 1264.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1264.20 | 1262.14 | 1264.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1262.50 | 1262.21 | 1263.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 14:45:00 | 1259.10 | 1262.17 | 1263.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 1266.90 | 1263.84 | 1264.11 | SL hit (close>static) qty=1.00 sl=1265.70 alert=retest2 |

### Cycle 153 — BUY (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 12:15:00 | 1269.90 | 1265.05 | 1264.64 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1258.60 | 1263.76 | 1264.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1252.80 | 1261.57 | 1263.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 1266.50 | 1261.38 | 1262.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1266.50 | 1261.38 | 1262.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1266.50 | 1261.38 | 1262.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1266.50 | 1261.38 | 1262.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1266.80 | 1262.46 | 1263.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1266.80 | 1262.46 | 1263.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1260.10 | 1261.99 | 1262.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 1265.00 | 1261.99 | 1262.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1262.70 | 1262.13 | 1262.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 1262.70 | 1262.13 | 1262.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1265.00 | 1262.70 | 1262.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 1265.00 | 1262.70 | 1262.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 1267.90 | 1263.74 | 1263.41 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 1256.10 | 1262.11 | 1262.88 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1276.80 | 1263.77 | 1262.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 1282.80 | 1272.12 | 1267.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1288.90 | 1296.46 | 1287.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1288.90 | 1296.46 | 1287.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1288.90 | 1296.46 | 1287.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1290.40 | 1296.46 | 1287.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1294.50 | 1296.07 | 1288.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1289.40 | 1296.07 | 1288.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1299.80 | 1308.93 | 1302.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:30:00 | 1315.00 | 1308.64 | 1305.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 1312.30 | 1309.82 | 1307.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 1312.10 | 1310.58 | 1308.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:45:00 | 1312.20 | 1310.84 | 1308.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1316.10 | 1316.73 | 1312.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1316.10 | 1316.73 | 1312.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1312.00 | 1318.48 | 1316.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 1313.50 | 1318.48 | 1316.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1313.40 | 1317.47 | 1315.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 1313.40 | 1317.47 | 1315.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1311.40 | 1315.73 | 1315.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 1313.00 | 1315.73 | 1315.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 1303.40 | 1313.26 | 1314.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 1303.40 | 1313.26 | 1314.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1302.00 | 1311.01 | 1313.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 1308.70 | 1307.51 | 1310.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:00:00 | 1308.70 | 1307.51 | 1310.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1307.00 | 1307.45 | 1309.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 1307.00 | 1307.45 | 1309.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1308.10 | 1307.58 | 1309.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:30:00 | 1311.00 | 1307.58 | 1309.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1305.30 | 1307.12 | 1309.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1301.90 | 1307.12 | 1309.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 1300.00 | 1304.86 | 1307.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 13:15:00 | 1236.81 | 1250.23 | 1263.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 13:15:00 | 1235.00 | 1250.23 | 1263.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1232.90 | 1231.89 | 1243.42 | SL hit (close>ema200) qty=0.50 sl=1231.89 alert=retest2 |

### Cycle 159 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 1247.90 | 1244.87 | 1244.54 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 1240.20 | 1243.93 | 1244.15 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 1247.80 | 1244.71 | 1244.48 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 1239.30 | 1244.10 | 1244.27 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 1245.50 | 1244.53 | 1244.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 1249.80 | 1246.81 | 1245.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 12:15:00 | 1248.90 | 1249.69 | 1247.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 12:15:00 | 1248.90 | 1249.69 | 1247.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1248.90 | 1249.69 | 1247.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 1249.00 | 1249.69 | 1247.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1248.90 | 1249.53 | 1247.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 1247.70 | 1249.53 | 1247.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1247.50 | 1249.13 | 1247.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:45:00 | 1248.10 | 1249.13 | 1247.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1248.00 | 1248.90 | 1247.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1242.70 | 1248.90 | 1247.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1244.00 | 1247.92 | 1247.40 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1239.10 | 1246.16 | 1246.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1235.10 | 1243.95 | 1245.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1259.20 | 1243.09 | 1244.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1259.20 | 1243.09 | 1244.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1259.20 | 1243.09 | 1244.03 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1259.90 | 1246.45 | 1245.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1266.70 | 1254.09 | 1249.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1251.50 | 1257.73 | 1253.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1251.50 | 1257.73 | 1253.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1251.50 | 1257.73 | 1253.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 1253.30 | 1257.73 | 1253.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1250.10 | 1256.20 | 1253.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1250.10 | 1256.20 | 1253.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1254.30 | 1255.82 | 1253.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 1258.90 | 1255.15 | 1253.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1240.80 | 1252.94 | 1253.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1240.80 | 1252.94 | 1253.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 1239.50 | 1247.10 | 1250.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 1236.30 | 1235.91 | 1240.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 11:30:00 | 1236.10 | 1235.91 | 1240.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1238.70 | 1236.47 | 1240.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 1238.70 | 1236.47 | 1240.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1237.50 | 1236.67 | 1239.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 1240.00 | 1236.67 | 1239.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1241.50 | 1237.64 | 1240.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 1241.50 | 1237.64 | 1240.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1240.00 | 1238.11 | 1240.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 1241.80 | 1238.11 | 1240.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1242.10 | 1238.91 | 1240.19 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 1249.90 | 1242.24 | 1241.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 12:15:00 | 1250.10 | 1243.81 | 1242.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 1276.10 | 1279.32 | 1268.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 10:00:00 | 1276.10 | 1279.32 | 1268.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1278.70 | 1279.68 | 1273.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:00:00 | 1288.60 | 1283.03 | 1278.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:30:00 | 1289.20 | 1285.05 | 1280.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 13:45:00 | 1288.20 | 1285.80 | 1281.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 1290.10 | 1284.34 | 1282.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1289.60 | 1286.92 | 1283.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 1287.00 | 1286.92 | 1283.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1286.30 | 1286.73 | 1284.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-29 13:15:00 | 1252.00 | 1279.81 | 1281.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 1252.00 | 1279.81 | 1281.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 1194.10 | 1256.03 | 1269.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 1199.80 | 1198.70 | 1210.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:30:00 | 1199.70 | 1198.70 | 1210.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1198.20 | 1198.06 | 1203.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 1194.50 | 1197.35 | 1202.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 1197.40 | 1197.64 | 1202.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 1209.00 | 1200.40 | 1202.61 | SL hit (close>static) qty=1.00 sl=1204.30 alert=retest2 |

### Cycle 169 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 1210.00 | 1204.32 | 1203.78 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 1199.00 | 1203.73 | 1203.77 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 1208.40 | 1203.05 | 1202.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 1212.80 | 1205.90 | 1204.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1229.90 | 1231.55 | 1224.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1229.90 | 1231.55 | 1224.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1237.60 | 1243.18 | 1239.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1237.60 | 1243.18 | 1239.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1235.00 | 1241.54 | 1239.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1235.80 | 1241.54 | 1239.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 1242.00 | 1242.22 | 1240.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 1244.60 | 1242.22 | 1240.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1239.50 | 1241.68 | 1240.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1239.50 | 1241.68 | 1240.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1238.30 | 1241.00 | 1240.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 1238.30 | 1241.00 | 1240.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1244.10 | 1245.60 | 1243.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1244.10 | 1245.60 | 1243.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1242.20 | 1244.92 | 1243.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:15:00 | 1245.80 | 1244.92 | 1243.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 1246.90 | 1245.32 | 1243.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 1248.60 | 1245.91 | 1243.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:00:00 | 1247.80 | 1246.13 | 1245.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:15:00 | 1248.00 | 1246.32 | 1245.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 14:00:00 | 1248.70 | 1246.80 | 1245.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1242.50 | 1245.94 | 1245.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 1242.50 | 1245.94 | 1245.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 1243.50 | 1245.45 | 1245.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1250.30 | 1245.45 | 1245.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1233.30 | 1245.31 | 1245.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 1233.30 | 1245.31 | 1245.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1226.20 | 1241.49 | 1244.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1242.60 | 1238.99 | 1242.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1242.60 | 1238.99 | 1242.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1242.60 | 1238.99 | 1242.33 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1249.60 | 1243.85 | 1243.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 1250.30 | 1245.14 | 1243.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 1244.20 | 1246.21 | 1244.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 1244.20 | 1246.21 | 1244.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1244.20 | 1246.21 | 1244.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1244.20 | 1246.21 | 1244.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1244.40 | 1245.85 | 1244.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 1246.70 | 1245.68 | 1244.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 1249.50 | 1246.45 | 1245.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:00:00 | 1246.50 | 1246.91 | 1245.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:30:00 | 1249.40 | 1247.47 | 1246.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1249.40 | 1251.99 | 1249.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1249.40 | 1251.99 | 1249.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1259.50 | 1253.49 | 1250.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 1260.50 | 1254.70 | 1251.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1260.30 | 1254.70 | 1251.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 1260.40 | 1256.12 | 1252.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 1264.70 | 1257.87 | 1253.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1272.70 | 1276.72 | 1271.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 1279.80 | 1275.55 | 1272.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:30:00 | 1279.00 | 1277.75 | 1274.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1266.40 | 1272.09 | 1273.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 1259.60 | 1253.06 | 1257.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 1259.60 | 1253.06 | 1257.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1259.60 | 1253.06 | 1257.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:15:00 | 1258.80 | 1253.06 | 1257.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1261.40 | 1254.73 | 1257.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 1265.60 | 1254.73 | 1257.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1275.20 | 1261.19 | 1260.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1277.30 | 1270.34 | 1265.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1268.00 | 1273.43 | 1269.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1268.00 | 1273.43 | 1269.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1268.00 | 1273.43 | 1269.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:00:00 | 1275.10 | 1273.50 | 1270.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:30:00 | 1275.80 | 1276.50 | 1273.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1277.10 | 1276.40 | 1273.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:15:00 | 1276.10 | 1274.51 | 1274.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1272.90 | 1274.76 | 1274.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:00:00 | 1272.90 | 1274.76 | 1274.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-17 13:15:00 | 1269.60 | 1273.73 | 1273.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 1269.60 | 1273.73 | 1273.93 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 1276.70 | 1273.81 | 1273.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 15:15:00 | 1280.00 | 1276.33 | 1275.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 11:15:00 | 1274.40 | 1278.37 | 1276.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 11:15:00 | 1274.40 | 1278.37 | 1276.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1274.40 | 1278.37 | 1276.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 1273.10 | 1278.37 | 1276.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1271.40 | 1276.97 | 1276.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 1271.40 | 1276.97 | 1276.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1280.10 | 1276.93 | 1276.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1283.80 | 1276.93 | 1276.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:45:00 | 1282.20 | 1278.09 | 1276.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 1281.60 | 1278.09 | 1276.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 1280.80 | 1279.02 | 1277.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1280.20 | 1282.03 | 1280.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 1280.20 | 1282.03 | 1280.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1279.70 | 1281.56 | 1279.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:15:00 | 1279.20 | 1281.56 | 1279.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1282.00 | 1281.65 | 1280.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 1283.50 | 1281.65 | 1280.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 15:00:00 | 1284.70 | 1282.26 | 1280.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1270.10 | 1267.12 | 1266.98 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1254.10 | 1265.99 | 1266.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 1250.40 | 1257.14 | 1261.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 1253.50 | 1252.56 | 1256.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 1253.50 | 1252.56 | 1256.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1257.60 | 1254.08 | 1256.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1251.60 | 1254.08 | 1256.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1255.00 | 1254.73 | 1256.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 1260.70 | 1255.92 | 1256.87 | SL hit (close>static) qty=1.00 sl=1257.60 alert=retest2 |

### Cycle 181 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1263.20 | 1255.02 | 1254.78 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 1244.60 | 1253.57 | 1254.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1229.80 | 1244.73 | 1249.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1210.80 | 1210.17 | 1218.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 1211.90 | 1210.17 | 1218.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1215.00 | 1211.96 | 1217.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1217.50 | 1211.96 | 1217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1215.00 | 1212.57 | 1217.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 1198.40 | 1212.57 | 1217.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1222.20 | 1173.61 | 1173.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 1222.20 | 1173.61 | 1173.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 1243.50 | 1214.20 | 1197.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 13:15:00 | 1236.00 | 1236.98 | 1224.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 14:00:00 | 1236.00 | 1236.98 | 1224.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1224.00 | 1233.60 | 1227.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1224.00 | 1233.60 | 1227.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1226.40 | 1232.16 | 1227.12 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1193.10 | 1219.70 | 1222.67 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1228.40 | 1218.34 | 1217.34 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1199.00 | 1216.44 | 1216.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1181.00 | 1209.35 | 1213.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1221.90 | 1193.10 | 1198.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1221.90 | 1193.10 | 1198.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1221.90 | 1193.10 | 1198.72 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1236.10 | 1207.87 | 1204.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1243.80 | 1226.99 | 1216.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1228.20 | 1240.89 | 1235.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 1228.20 | 1240.89 | 1235.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1228.20 | 1240.89 | 1235.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1228.20 | 1240.89 | 1235.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1231.40 | 1238.99 | 1235.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 1226.00 | 1238.99 | 1235.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1230.40 | 1236.36 | 1234.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 1228.30 | 1236.36 | 1234.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1240.90 | 1237.02 | 1235.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:30:00 | 1236.30 | 1237.02 | 1235.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1261.80 | 1261.36 | 1255.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 1257.50 | 1261.36 | 1255.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1255.90 | 1260.27 | 1255.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 1255.00 | 1260.27 | 1255.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 1255.70 | 1259.35 | 1255.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1261.20 | 1259.35 | 1255.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 1265.70 | 1267.81 | 1267.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 10:15:00 | 1265.70 | 1267.81 | 1267.89 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1269.80 | 1268.21 | 1268.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1277.80 | 1270.13 | 1268.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1282.90 | 1284.70 | 1281.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 1282.90 | 1284.70 | 1281.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1277.60 | 1283.28 | 1280.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1282.50 | 1283.28 | 1280.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1285.00 | 1283.62 | 1281.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:00:00 | 1296.80 | 1285.93 | 1283.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1294.00 | 1287.54 | 1284.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 1299.90 | 1289.71 | 1286.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 10:15:00 | 1294.20 | 1299.00 | 1296.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1300.60 | 1304.68 | 1300.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 1300.60 | 1304.68 | 1300.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 1307.00 | 1305.14 | 1300.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1313.40 | 1305.17 | 1301.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1297.00 | 1310.34 | 1307.33 | SL hit (close<static) qty=1.00 sl=1300.40 alert=retest2 |

### Cycle 190 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 1283.90 | 1302.35 | 1304.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1276.60 | 1290.51 | 1296.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1293.80 | 1286.73 | 1292.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 1293.80 | 1286.73 | 1292.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1293.80 | 1286.73 | 1292.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 1293.50 | 1286.73 | 1292.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1288.10 | 1287.00 | 1291.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1281.70 | 1287.00 | 1291.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:45:00 | 1278.30 | 1285.94 | 1290.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:15:00 | 1282.60 | 1286.37 | 1290.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 13:00:00 | 1287.60 | 1286.16 | 1289.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1290.70 | 1287.07 | 1289.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 1290.70 | 1287.07 | 1289.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 1289.40 | 1287.53 | 1289.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:45:00 | 1298.70 | 1287.53 | 1289.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 1291.30 | 1288.29 | 1289.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 1303.10 | 1288.29 | 1289.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1306.20 | 1291.87 | 1291.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 1306.20 | 1291.87 | 1291.38 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 1284.00 | 1298.04 | 1299.35 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1317.10 | 1299.98 | 1298.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 1319.30 | 1303.84 | 1300.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1310.30 | 1319.93 | 1314.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1310.30 | 1319.93 | 1314.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1310.30 | 1319.93 | 1314.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 1327.60 | 1319.78 | 1315.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1296.50 | 1316.18 | 1315.91 | SL hit (close<static) qty=1.00 sl=1303.60 alert=retest2 |

### Cycle 194 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1296.00 | 1312.14 | 1314.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1283.30 | 1304.34 | 1310.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1280.50 | 1279.26 | 1289.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:15:00 | 1279.60 | 1279.26 | 1289.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1284.80 | 1280.10 | 1285.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:30:00 | 1285.00 | 1280.10 | 1285.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1281.90 | 1280.46 | 1285.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1291.50 | 1280.46 | 1285.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1292.50 | 1282.87 | 1286.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1293.80 | 1282.87 | 1286.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1298.40 | 1285.97 | 1287.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 1298.40 | 1285.97 | 1287.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1294.50 | 1288.96 | 1288.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1296.60 | 1290.49 | 1289.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 1286.90 | 1290.86 | 1289.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 15:15:00 | 1286.90 | 1290.86 | 1289.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1286.90 | 1290.86 | 1289.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 1272.50 | 1290.86 | 1289.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1276.30 | 1287.95 | 1288.40 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 1295.90 | 1286.69 | 1285.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 1302.30 | 1291.70 | 1288.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 1270.10 | 1289.04 | 1287.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 1270.10 | 1289.04 | 1287.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1270.10 | 1289.04 | 1287.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1270.10 | 1289.04 | 1287.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 1279.80 | 1287.19 | 1287.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 13:15:00 | 1267.50 | 1280.38 | 1283.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 1291.50 | 1266.05 | 1270.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 1291.50 | 1266.05 | 1270.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1291.50 | 1266.05 | 1270.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1291.50 | 1266.05 | 1270.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1291.90 | 1271.22 | 1272.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 1291.90 | 1271.22 | 1272.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1297.30 | 1276.44 | 1274.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 1299.40 | 1281.03 | 1276.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 12:15:00 | 1292.40 | 1295.50 | 1288.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:00:00 | 1292.40 | 1295.50 | 1288.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1279.40 | 1291.37 | 1287.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1279.40 | 1291.37 | 1287.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1281.10 | 1289.32 | 1286.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1265.00 | 1289.32 | 1286.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1260.40 | 1283.53 | 1284.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 1252.00 | 1265.03 | 1273.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1266.00 | 1265.23 | 1272.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1266.00 | 1265.23 | 1272.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1266.00 | 1265.23 | 1272.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1256.10 | 1265.23 | 1272.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1193.29 | 1222.89 | 1245.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 1218.80 | 1217.21 | 1233.42 | SL hit (close>ema200) qty=0.50 sl=1217.21 alert=retest2 |

### Cycle 201 — BUY (started 2026-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 12:15:00 | 1212.80 | 1202.72 | 1202.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 1222.20 | 1209.80 | 1206.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1212.90 | 1223.08 | 1216.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1212.90 | 1223.08 | 1216.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1212.90 | 1223.08 | 1216.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:45:00 | 1213.80 | 1223.08 | 1216.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1226.50 | 1223.77 | 1217.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1229.40 | 1225.06 | 1219.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 1216.00 | 1219.80 | 1219.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 14:15:00 | 1216.00 | 1219.80 | 1219.98 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 1222.20 | 1220.35 | 1220.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 13:15:00 | 1223.90 | 1221.60 | 1220.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 1221.20 | 1221.52 | 1220.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 14:15:00 | 1221.20 | 1221.52 | 1220.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 1221.20 | 1221.52 | 1220.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 1221.20 | 1221.52 | 1220.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 1220.80 | 1221.38 | 1220.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 1218.70 | 1221.38 | 1220.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1223.40 | 1221.78 | 1221.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 11:45:00 | 1225.50 | 1222.90 | 1221.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 1224.60 | 1229.72 | 1229.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 1220.00 | 1227.77 | 1228.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1220.00 | 1227.77 | 1228.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1212.80 | 1221.54 | 1224.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 12:15:00 | 1225.30 | 1221.73 | 1223.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 12:15:00 | 1225.30 | 1221.73 | 1223.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 1225.30 | 1221.73 | 1223.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 1225.30 | 1221.73 | 1223.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 1219.00 | 1221.19 | 1223.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 1214.80 | 1219.91 | 1222.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1306.10 | 1236.71 | 1229.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1306.10 | 1236.71 | 1229.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 1321.60 | 1272.08 | 1249.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 15:15:00 | 1313.00 | 1314.51 | 1293.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 1319.30 | 1314.51 | 1293.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1342.70 | 1347.75 | 1334.46 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1317.70 | 1331.53 | 1332.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 1301.50 | 1321.49 | 1326.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1283.00 | 1280.17 | 1292.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:15:00 | 1286.40 | 1280.17 | 1292.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1283.20 | 1280.78 | 1292.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:30:00 | 1289.60 | 1280.78 | 1292.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1295.10 | 1283.64 | 1292.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:45:00 | 1297.50 | 1283.64 | 1292.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1311.30 | 1289.17 | 1294.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 1311.30 | 1289.17 | 1294.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1325.00 | 1296.34 | 1296.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:00:00 | 1325.00 | 1296.34 | 1296.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1310.00 | 1299.07 | 1298.03 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 1294.40 | 1300.60 | 1301.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 1292.50 | 1298.93 | 1300.47 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-25 12:15:00 | 900.61 | 2023-05-30 15:15:00 | 901.88 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2023-05-25 15:00:00 | 901.78 | 2023-05-30 15:15:00 | 901.88 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2023-06-08 15:15:00 | 931.70 | 2023-06-30 09:15:00 | 1024.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-09 15:00:00 | 934.19 | 2023-06-30 09:15:00 | 1027.61 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-07-12 12:15:00 | 1033.12 | 2023-07-12 13:15:00 | 1038.20 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-07-24 10:30:00 | 1066.79 | 2023-08-07 11:15:00 | 1129.20 | STOP_HIT | 1.00 | 5.85% |
| BUY | retest2 | 2023-07-24 12:15:00 | 1069.07 | 2023-08-07 11:15:00 | 1129.20 | STOP_HIT | 1.00 | 5.62% |
| BUY | retest2 | 2023-08-21 09:30:00 | 1176.80 | 2023-08-25 09:15:00 | 1166.06 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-08-21 10:30:00 | 1175.93 | 2023-08-25 09:15:00 | 1166.06 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-09-06 11:30:00 | 1118.80 | 2023-09-07 09:15:00 | 1128.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-09-22 10:15:00 | 1104.08 | 2023-09-29 10:15:00 | 1121.33 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2023-09-22 13:30:00 | 1103.60 | 2023-09-29 10:15:00 | 1121.33 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-09-25 09:15:00 | 1094.49 | 2023-09-29 10:15:00 | 1121.33 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2023-09-27 10:45:00 | 1105.42 | 2023-09-29 10:15:00 | 1121.33 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-09-28 09:15:00 | 1095.00 | 2023-09-29 10:15:00 | 1121.33 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2023-09-28 11:15:00 | 1097.00 | 2023-09-29 10:15:00 | 1121.33 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2023-10-05 12:00:00 | 1085.46 | 2023-10-09 09:15:00 | 1100.01 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-10-06 11:15:00 | 1084.90 | 2023-10-09 09:15:00 | 1100.01 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-10-13 11:00:00 | 1102.83 | 2023-10-13 11:15:00 | 1096.91 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2023-10-25 14:15:00 | 1107.46 | 2023-10-30 09:15:00 | 1052.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-25 14:15:00 | 1107.46 | 2023-10-31 09:15:00 | 1082.24 | STOP_HIT | 0.50 | 2.28% |
| BUY | retest1 | 2023-11-09 09:15:00 | 1093.58 | 2023-11-10 09:15:00 | 1081.99 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-11-24 09:15:00 | 1146.60 | 2023-12-06 12:15:00 | 1143.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2023-11-28 09:15:00 | 1140.85 | 2023-12-06 12:15:00 | 1143.40 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2023-11-28 12:00:00 | 1138.00 | 2023-12-06 12:15:00 | 1143.40 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2023-11-28 14:30:00 | 1139.07 | 2023-12-06 12:15:00 | 1143.40 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2023-11-29 09:15:00 | 1135.54 | 2023-12-06 12:15:00 | 1143.40 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2023-11-29 10:00:00 | 1136.55 | 2023-12-06 12:15:00 | 1143.40 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2023-11-29 11:15:00 | 1137.77 | 2023-12-06 12:15:00 | 1143.40 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2023-12-12 14:45:00 | 1105.69 | 2023-12-14 11:15:00 | 1120.10 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-01-08 10:15:00 | 1153.11 | 2024-01-15 11:15:00 | 1158.30 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-01-09 11:15:00 | 1156.18 | 2024-01-15 11:15:00 | 1158.30 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-01-09 13:30:00 | 1156.41 | 2024-01-15 11:15:00 | 1158.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-01-11 09:15:00 | 1139.21 | 2024-01-15 11:15:00 | 1158.30 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-01-19 11:45:00 | 1127.47 | 2024-01-23 09:15:00 | 1147.64 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-01-19 14:00:00 | 1127.88 | 2024-01-23 09:15:00 | 1147.64 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-01-20 13:30:00 | 1127.80 | 2024-01-23 09:15:00 | 1147.64 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-01-20 14:00:00 | 1127.59 | 2024-01-23 09:15:00 | 1147.64 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-01-29 13:00:00 | 1162.92 | 2024-02-13 09:15:00 | 1279.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-29 13:30:00 | 1161.78 | 2024-02-13 09:15:00 | 1277.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-29 14:15:00 | 1165.28 | 2024-02-13 09:15:00 | 1281.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-20 12:30:00 | 1277.60 | 2024-02-21 11:15:00 | 1267.67 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-02-20 13:00:00 | 1277.75 | 2024-02-21 11:15:00 | 1267.67 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-03-14 09:15:00 | 1252.50 | 2024-03-18 09:15:00 | 1266.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-03-14 15:15:00 | 1257.02 | 2024-03-18 09:15:00 | 1266.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-04-12 12:45:00 | 1230.47 | 2024-04-22 13:15:00 | 1207.98 | STOP_HIT | 1.00 | 1.83% |
| BUY | retest2 | 2024-05-02 09:15:00 | 1249.07 | 2024-05-07 11:15:00 | 1249.35 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-05-08 09:15:00 | 1208.74 | 2024-05-16 13:15:00 | 1148.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 09:15:00 | 1208.74 | 2024-05-16 14:15:00 | 1174.00 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2024-06-03 10:30:00 | 1159.62 | 2024-06-05 09:15:00 | 1167.52 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-06-03 12:15:00 | 1160.00 | 2024-06-05 14:15:00 | 1166.67 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-06-03 13:15:00 | 1159.92 | 2024-06-05 14:15:00 | 1166.67 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-06-04 10:15:00 | 1160.00 | 2024-06-05 14:15:00 | 1166.67 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-06-04 12:00:00 | 1132.10 | 2024-06-05 14:15:00 | 1166.67 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-06-14 14:30:00 | 1216.12 | 2024-06-18 09:15:00 | 1201.60 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-06-26 11:15:00 | 1215.96 | 2024-07-12 13:15:00 | 1337.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-26 12:15:00 | 1215.99 | 2024-07-12 13:15:00 | 1337.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-26 15:00:00 | 1215.63 | 2024-07-12 13:15:00 | 1337.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-27 09:15:00 | 1240.05 | 2024-07-15 11:15:00 | 1364.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-27 15:00:00 | 1248.12 | 2024-07-15 13:15:00 | 1372.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-28 09:15:00 | 1272.00 | 2024-07-18 09:15:00 | 1327.60 | STOP_HIT | 1.00 | 4.37% |
| BUY | retest2 | 2024-07-25 10:30:00 | 1361.79 | 2024-07-30 12:15:00 | 1359.91 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-07-26 14:30:00 | 1364.00 | 2024-07-30 12:15:00 | 1359.91 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-07-29 09:30:00 | 1363.39 | 2024-07-30 12:15:00 | 1359.91 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1400.40 | 2024-08-12 11:15:00 | 1382.48 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-08-23 11:15:00 | 1394.00 | 2024-08-23 12:15:00 | 1391.89 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-09-05 11:15:00 | 1354.80 | 2024-09-13 14:15:00 | 1332.09 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2024-09-05 12:15:00 | 1356.33 | 2024-09-13 14:15:00 | 1332.09 | STOP_HIT | 1.00 | 1.79% |
| SELL | retest2 | 2024-09-17 10:15:00 | 1325.92 | 2024-09-17 13:15:00 | 1331.69 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-09-17 11:00:00 | 1326.08 | 2024-09-17 13:15:00 | 1331.69 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-09-17 11:30:00 | 1324.79 | 2024-09-17 13:15:00 | 1331.69 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-09-17 13:45:00 | 1326.15 | 2024-09-23 10:15:00 | 1322.46 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2024-09-18 09:15:00 | 1324.71 | 2024-09-23 10:15:00 | 1322.46 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-09-18 09:45:00 | 1324.13 | 2024-09-23 10:15:00 | 1322.46 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2024-09-23 09:30:00 | 1323.06 | 2024-09-23 10:15:00 | 1322.46 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-09-25 15:00:00 | 1339.12 | 2024-10-04 09:15:00 | 1338.51 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-10-07 12:45:00 | 1328.00 | 2024-10-09 11:15:00 | 1340.08 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-10-08 14:15:00 | 1328.52 | 2024-10-09 11:15:00 | 1340.08 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-10-17 12:15:00 | 1341.00 | 2024-10-22 10:15:00 | 1338.45 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-10-17 15:00:00 | 1341.65 | 2024-10-22 10:15:00 | 1338.45 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-10-18 09:45:00 | 1345.20 | 2024-10-22 10:15:00 | 1338.45 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-10-21 10:30:00 | 1341.00 | 2024-10-22 10:15:00 | 1338.45 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-10-29 09:30:00 | 1300.75 | 2024-10-30 09:15:00 | 1235.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-29 09:30:00 | 1300.75 | 2024-10-31 11:15:00 | 1260.70 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2024-11-27 11:00:00 | 1199.05 | 2024-12-02 09:15:00 | 1221.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-11-27 13:45:00 | 1200.50 | 2024-12-02 09:15:00 | 1221.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-11-27 15:00:00 | 1200.05 | 2024-12-02 09:15:00 | 1221.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-11-28 09:30:00 | 1198.15 | 2024-12-02 09:15:00 | 1221.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-12-04 13:30:00 | 1218.60 | 2024-12-10 14:15:00 | 1240.10 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2024-12-04 15:15:00 | 1216.50 | 2024-12-10 14:15:00 | 1240.10 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2024-12-05 10:00:00 | 1219.45 | 2024-12-10 14:15:00 | 1240.10 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2024-12-13 09:15:00 | 1237.00 | 2024-12-13 12:15:00 | 1244.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-12-27 10:45:00 | 1384.60 | 2025-01-01 11:15:00 | 1363.45 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-12-30 12:30:00 | 1388.30 | 2025-01-01 11:15:00 | 1363.45 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-30 14:00:00 | 1383.65 | 2025-01-01 11:15:00 | 1363.45 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-12-30 14:45:00 | 1385.30 | 2025-01-01 11:15:00 | 1363.45 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-12-31 10:45:00 | 1385.70 | 2025-01-01 11:15:00 | 1363.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-01-16 10:15:00 | 1324.50 | 2025-01-24 09:15:00 | 1258.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 10:15:00 | 1324.50 | 2025-01-27 14:15:00 | 1192.05 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-07 11:30:00 | 1239.80 | 2025-02-10 09:15:00 | 1217.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-02-07 13:30:00 | 1240.90 | 2025-02-10 09:15:00 | 1217.20 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-02-07 15:00:00 | 1240.55 | 2025-02-10 09:15:00 | 1217.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1185.35 | 2025-02-25 14:15:00 | 1126.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1185.35 | 2025-03-03 12:15:00 | 1115.80 | STOP_HIT | 0.50 | 5.87% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1149.20 | 2025-03-05 11:15:00 | 1126.50 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest1 | 2025-03-13 09:15:00 | 1103.00 | 2025-03-17 09:15:00 | 1144.30 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-04-01 12:45:00 | 1152.65 | 2025-04-03 09:15:00 | 1170.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1079.80 | 2025-04-11 09:15:00 | 1125.70 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2025-04-23 09:15:00 | 1179.70 | 2025-04-25 12:15:00 | 1174.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-05-05 09:30:00 | 1174.70 | 2025-05-12 09:15:00 | 1183.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-05-05 10:15:00 | 1172.90 | 2025-05-12 09:15:00 | 1183.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1155.30 | 2025-05-12 09:15:00 | 1183.20 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1226.00 | 2025-05-22 09:15:00 | 1214.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-05-19 12:15:00 | 1217.30 | 2025-05-22 09:15:00 | 1214.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-05-20 09:15:00 | 1222.30 | 2025-05-22 09:15:00 | 1214.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2025-05-28 09:15:00 | 1247.40 | 2025-05-28 12:15:00 | 1238.70 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2025-05-28 10:30:00 | 1246.40 | 2025-05-28 12:15:00 | 1238.70 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1244.80 | 2025-06-03 10:15:00 | 1245.70 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-05-29 10:45:00 | 1247.80 | 2025-06-03 10:15:00 | 1245.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-05-30 10:30:00 | 1245.70 | 2025-06-03 10:15:00 | 1245.70 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-05-30 11:00:00 | 1248.30 | 2025-06-03 10:15:00 | 1245.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1288.60 | 2025-06-16 14:15:00 | 1347.00 | STOP_HIT | 1.00 | 4.53% |
| SELL | retest2 | 2025-06-19 14:30:00 | 1322.80 | 2025-06-19 15:15:00 | 1331.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-20 10:45:00 | 1323.00 | 2025-06-24 09:15:00 | 1335.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-20 11:15:00 | 1321.90 | 2025-06-24 09:15:00 | 1335.80 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1320.70 | 2025-06-24 09:15:00 | 1335.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1312.50 | 2025-06-24 09:15:00 | 1335.80 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest1 | 2025-07-02 09:15:00 | 1275.50 | 2025-07-03 10:15:00 | 1281.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-07-16 09:15:00 | 1256.40 | 2025-07-16 15:15:00 | 1259.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-16 13:45:00 | 1257.20 | 2025-07-16 15:15:00 | 1259.50 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1244.90 | 2025-07-24 09:15:00 | 1276.90 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-07-22 11:15:00 | 1242.50 | 2025-07-24 09:15:00 | 1276.90 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-07-22 13:45:00 | 1242.70 | 2025-07-24 09:15:00 | 1276.90 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-07-30 11:15:00 | 1292.00 | 2025-07-31 09:15:00 | 1266.50 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-30 14:45:00 | 1289.50 | 2025-07-31 09:15:00 | 1266.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-08 10:45:00 | 1197.60 | 2025-08-08 14:15:00 | 1210.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-12 10:30:00 | 1215.80 | 2025-08-20 09:15:00 | 1241.40 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-08-12 11:15:00 | 1215.60 | 2025-08-20 09:15:00 | 1241.40 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest1 | 2025-08-25 09:30:00 | 1284.10 | 2025-08-26 09:15:00 | 1264.20 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-09-03 14:45:00 | 1259.10 | 2025-09-04 11:15:00 | 1266.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-17 09:30:00 | 1315.00 | 2025-09-22 13:15:00 | 1303.40 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-17 12:30:00 | 1312.30 | 2025-09-22 13:15:00 | 1303.40 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-18 09:15:00 | 1312.10 | 2025-09-22 13:15:00 | 1303.40 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-18 10:45:00 | 1312.20 | 2025-09-22 13:15:00 | 1303.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1301.90 | 2025-09-29 13:15:00 | 1236.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 12:00:00 | 1300.00 | 2025-09-29 13:15:00 | 1235.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1301.90 | 2025-10-01 09:15:00 | 1232.90 | STOP_HIT | 0.50 | 5.30% |
| SELL | retest2 | 2025-09-24 12:00:00 | 1300.00 | 2025-10-01 09:15:00 | 1232.90 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2025-10-13 14:15:00 | 1258.90 | 2025-10-14 10:15:00 | 1240.80 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-27 12:00:00 | 1288.60 | 2025-10-29 13:15:00 | 1252.00 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-10-27 12:30:00 | 1289.20 | 2025-10-29 13:15:00 | 1252.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-10-27 13:45:00 | 1288.20 | 2025-10-29 13:15:00 | 1252.00 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-10-28 15:00:00 | 1290.10 | 2025-10-29 13:15:00 | 1252.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-11-06 10:00:00 | 1194.50 | 2025-11-06 12:15:00 | 1209.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-11-06 11:15:00 | 1197.40 | 2025-11-06 12:15:00 | 1209.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-11-20 12:30:00 | 1248.60 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-11-21 12:00:00 | 1247.80 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-21 13:15:00 | 1248.00 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-21 14:00:00 | 1248.70 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-11-24 09:15:00 | 1250.30 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-11-27 13:45:00 | 1246.70 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-11-27 15:00:00 | 1249.50 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 1.53% |
| BUY | retest2 | 2025-11-28 11:00:00 | 1246.50 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2025-11-28 11:30:00 | 1249.40 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-12-01 11:45:00 | 1260.50 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-12-01 12:15:00 | 1260.30 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1260.40 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-12-02 09:15:00 | 1264.70 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-12-04 15:15:00 | 1279.80 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-05 10:30:00 | 1279.00 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-15 12:00:00 | 1275.10 | 2025-12-17 13:15:00 | 1269.60 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-16 09:30:00 | 1275.80 | 2025-12-17 13:15:00 | 1269.60 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-16 10:30:00 | 1277.10 | 2025-12-17 13:15:00 | 1269.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-17 11:15:00 | 1276.10 | 2025-12-17 13:15:00 | 1269.60 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-22 09:15:00 | 1283.80 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-22 10:45:00 | 1282.20 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-12-22 11:15:00 | 1281.60 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-12-22 12:30:00 | 1280.80 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-12-23 14:15:00 | 1283.50 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-23 15:00:00 | 1284.70 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1251.60 | 2026-01-05 11:15:00 | 1260.70 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-05 11:15:00 | 1255.00 | 2026-01-05 11:15:00 | 1260.70 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-05 11:45:00 | 1255.30 | 2026-01-05 12:15:00 | 1260.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-01-05 13:45:00 | 1255.10 | 2026-01-07 09:15:00 | 1263.20 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-01-06 12:15:00 | 1245.80 | 2026-01-07 09:15:00 | 1263.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-13 09:15:00 | 1198.40 | 2026-01-22 09:15:00 | 1222.20 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1261.20 | 2026-02-17 10:15:00 | 1265.70 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2026-02-23 10:00:00 | 1296.80 | 2026-02-27 09:15:00 | 1297.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2026-02-23 11:00:00 | 1294.00 | 2026-02-27 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-02-23 13:15:00 | 1299.90 | 2026-02-27 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-02-25 10:15:00 | 1294.20 | 2026-02-27 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-26 09:15:00 | 1313.40 | 2026-02-27 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1281.70 | 2026-03-05 09:15:00 | 1306.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-03-04 09:45:00 | 1278.30 | 2026-03-05 09:15:00 | 1306.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-03-04 11:15:00 | 1282.60 | 2026-03-05 09:15:00 | 1306.20 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-03-04 13:00:00 | 1287.60 | 2026-03-05 09:15:00 | 1306.20 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-03-12 12:15:00 | 1327.60 | 2026-03-13 09:15:00 | 1296.50 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1256.10 | 2026-04-02 09:15:00 | 1193.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1256.10 | 2026-04-02 14:15:00 | 1218.80 | STOP_HIT | 0.50 | 2.97% |
| BUY | retest2 | 2026-04-13 12:30:00 | 1229.40 | 2026-04-15 14:15:00 | 1216.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-04-17 11:45:00 | 1225.50 | 2026-04-21 11:15:00 | 1220.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-04-21 10:30:00 | 1224.60 | 2026-04-21 11:15:00 | 1220.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-04-22 15:00:00 | 1214.80 | 2026-04-23 09:15:00 | 1306.10 | STOP_HIT | 1.00 | -7.52% |
