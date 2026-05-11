# Kfin Technologies Ltd. (KFINTECH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 917.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 2 |
| TARGET_HIT | 5 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 5 / 13 / 2
- **Avg / median % per leg:** 0.66% / -1.68%
- **Sum % (uncompounded):** 13.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| SELL (all) | 16 | 5 | 31.2% | 1 | 13 | 2 | -1.67% | -26.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 5 | 31.2% | 1 | 13 | 2 | -1.67% | -26.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 9 | 45.0% | 5 | 13 | 2 | 0.66% | 13.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 1052.95 | 1209.61 | 1209.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 14:15:00 | 1038.40 | 1184.56 | 1195.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 969.95 | 957.29 | 1030.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 09:45:00 | 973.20 | 957.29 | 1030.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 1028.65 | 959.84 | 1028.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:00:00 | 1028.65 | 959.84 | 1028.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1045.05 | 960.69 | 1028.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:00:00 | 1045.05 | 960.69 | 1028.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 1031.55 | 961.40 | 1028.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 1027.25 | 995.88 | 1038.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:30:00 | 1027.10 | 1000.18 | 1037.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 998.35 | 1000.79 | 1037.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:30:00 | 1020.75 | 1001.29 | 1037.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 1037.70 | 1002.05 | 1037.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 1037.70 | 1002.05 | 1037.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 1030.25 | 1002.33 | 1037.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:30:00 | 1038.00 | 1002.33 | 1037.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 1035.00 | 1002.66 | 1037.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 1027.90 | 1002.66 | 1037.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 1044.50 | 1003.36 | 1037.42 | SL hit (close>static) qty=1.00 sl=1038.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 1247.00 | 1052.71 | 1052.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 1282.10 | 1058.77 | 1055.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 09:15:00 | 1098.00 | 1113.68 | 1087.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:15:00 | 1089.10 | 1113.68 | 1087.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1095.20 | 1113.50 | 1087.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 1089.90 | 1113.50 | 1087.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1088.90 | 1113.02 | 1087.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 1088.90 | 1113.02 | 1087.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1086.60 | 1112.52 | 1087.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 1086.60 | 1112.52 | 1087.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1092.00 | 1112.32 | 1087.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 1077.70 | 1112.32 | 1087.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1065.50 | 1111.85 | 1087.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 1065.50 | 1111.85 | 1087.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1059.00 | 1111.33 | 1086.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 1059.00 | 1111.33 | 1086.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1131.50 | 1107.12 | 1086.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 1160.80 | 1084.67 | 1080.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:15:00 | 1160.40 | 1085.41 | 1080.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1161.20 | 1088.82 | 1082.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:15:00 | 1162.30 | 1090.21 | 1083.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-10 10:15:00 | 1276.88 | 1121.89 | 1101.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1093.00 | 1193.90 | 1194.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 1081.00 | 1192.78 | 1193.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1116.10 | 1110.36 | 1138.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 1116.10 | 1110.36 | 1138.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1138.70 | 1109.95 | 1132.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1138.70 | 1109.95 | 1132.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1144.50 | 1110.30 | 1132.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 1143.50 | 1110.30 | 1132.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1141.40 | 1111.50 | 1132.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 1141.40 | 1111.50 | 1132.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1135.00 | 1114.78 | 1133.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 1135.20 | 1114.78 | 1133.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 1142.80 | 1115.06 | 1133.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 1142.80 | 1115.06 | 1133.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1112.00 | 1083.05 | 1107.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 1112.50 | 1083.05 | 1107.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1121.50 | 1083.43 | 1107.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1121.50 | 1083.43 | 1107.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1109.50 | 1088.32 | 1108.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 1110.70 | 1088.32 | 1108.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1120.60 | 1088.95 | 1108.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1121.00 | 1088.95 | 1108.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1118.20 | 1112.91 | 1116.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:00:00 | 1116.40 | 1113.57 | 1117.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 1060.58 | 1109.05 | 1114.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 1112.20 | 1105.06 | 1111.98 | SL hit (close>ema200) qty=0.50 sl=1105.06 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-03-27 09:15:00 | 1027.25 | 2025-04-02 10:15:00 | 1044.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-03-28 13:30:00 | 1027.10 | 2025-04-03 09:15:00 | 1071.70 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2025-04-01 09:15:00 | 998.35 | 2025-04-03 09:15:00 | 1071.70 | STOP_HIT | 1.00 | -7.35% |
| SELL | retest2 | 2025-04-01 10:30:00 | 1020.75 | 2025-04-03 09:15:00 | 1071.70 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1027.90 | 2025-04-03 09:15:00 | 1071.70 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2025-04-07 09:15:00 | 987.15 | 2025-04-07 09:15:00 | 888.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 09:45:00 | 1029.30 | 2025-04-16 12:15:00 | 1059.20 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1024.30 | 2025-04-16 12:15:00 | 1059.20 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-04-11 10:30:00 | 1011.10 | 2025-04-16 12:15:00 | 1059.20 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2025-04-11 13:15:00 | 1010.65 | 2025-04-16 12:15:00 | 1059.20 | STOP_HIT | 1.00 | -4.80% |
| SELL | retest2 | 2025-04-15 09:45:00 | 1010.80 | 2025-04-16 12:15:00 | 1059.20 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2025-04-15 10:15:00 | 1010.80 | 2025-04-16 12:15:00 | 1059.20 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2025-06-03 09:15:00 | 1160.80 | 2025-06-10 10:15:00 | 1276.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-03 10:15:00 | 1160.40 | 2025-06-10 10:15:00 | 1276.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 09:15:00 | 1161.20 | 2025-06-10 10:15:00 | 1277.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 11:15:00 | 1162.30 | 2025-06-10 10:15:00 | 1278.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-30 10:00:00 | 1116.40 | 2025-11-06 09:15:00 | 1060.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 10:00:00 | 1116.40 | 2025-11-07 14:15:00 | 1112.20 | STOP_HIT | 0.50 | 0.38% |
| SELL | retest2 | 2025-11-12 10:00:00 | 1113.00 | 2025-11-24 14:15:00 | 1057.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 10:00:00 | 1113.00 | 2025-12-01 10:15:00 | 1102.10 | STOP_HIT | 0.50 | 0.98% |
