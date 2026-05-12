# Signatureglobal (India) Ltd. (SIGNATURE)

## Backtest Summary

- **Window:** 2023-09-27 09:15:00 → 2026-05-11 15:15:00 (4512 bars)
- **Last close:** 885.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 21 |
| PARTIAL | 9 |
| TARGET_HIT | 7 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 16
- **Target hits / Stop hits / Partials:** 7 / 18 / 9
- **Avg / median % per leg:** 2.33% / 0.64%
- **Sum % (uncompounded):** 79.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.75% | -21.1% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.53% | -12.2% |
| SELL (all) | 22 | 18 | 81.8% | 7 | 6 | 9 | 4.57% | 100.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 18 | 81.8% | 7 | 6 | 9 | 4.57% | 100.4% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| retest2 (combined) | 30 | 18 | 60.0% | 7 | 14 | 9 | 2.94% | 88.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 1363.50 | 1476.80 | 1476.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1350.80 | 1457.00 | 1466.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 1361.75 | 1354.91 | 1394.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 09:30:00 | 1366.65 | 1354.91 | 1394.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1368.05 | 1356.18 | 1392.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:30:00 | 1390.95 | 1356.18 | 1392.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1343.00 | 1304.17 | 1344.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 14:00:00 | 1325.05 | 1308.03 | 1344.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 1324.05 | 1308.57 | 1344.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 14:15:00 | 1364.15 | 1309.87 | 1344.24 | SL hit (close>static) qty=1.00 sl=1345.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1232.20 | 1149.88 | 1149.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 10:15:00 | 1261.60 | 1157.27 | 1153.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1234.00 | 1240.71 | 1210.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 14:30:00 | 1249.10 | 1240.52 | 1211.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 11:45:00 | 1252.50 | 1240.59 | 1212.03 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 14:00:00 | 1254.70 | 1240.73 | 1213.36 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 10:00:00 | 1250.30 | 1241.06 | 1213.93 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1230.20 | 1246.26 | 1229.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 1230.20 | 1246.26 | 1229.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1224.10 | 1246.04 | 1229.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 1224.10 | 1246.04 | 1229.88 | SL hit (close<ema400) qty=1.00 sl=1229.88 alert=retest1 |

### Cycle 3 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1113.60 | 1217.56 | 1217.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 1103.80 | 1190.84 | 1203.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 1124.30 | 1123.81 | 1151.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 12:00:00 | 1124.30 | 1123.81 | 1151.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1153.10 | 1124.92 | 1148.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1153.10 | 1124.92 | 1148.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1139.80 | 1125.07 | 1148.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 1136.70 | 1125.07 | 1148.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 1134.00 | 1125.16 | 1148.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 1137.10 | 1125.70 | 1148.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 1138.80 | 1126.35 | 1147.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1139.10 | 1127.34 | 1147.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:45:00 | 1154.30 | 1127.34 | 1147.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1150.00 | 1127.57 | 1147.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1151.90 | 1127.78 | 1147.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1145.90 | 1127.96 | 1147.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 1138.30 | 1128.06 | 1147.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1079.87 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1077.30 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1080.24 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1081.86 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:15:00 | 1081.38 | 1123.46 | 1142.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-07 10:15:00 | 1023.03 | 1096.92 | 1123.80 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 1126.70 | 1099.28 | 1099.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 1134.00 | 1104.18 | 1101.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 1104.30 | 1104.91 | 1102.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 1104.30 | 1104.91 | 1102.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 1104.30 | 1104.91 | 1102.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 10:45:00 | 1113.50 | 1104.98 | 1102.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 15:00:00 | 1111.40 | 1106.02 | 1102.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 1113.40 | 1106.13 | 1103.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 13:30:00 | 1111.30 | 1106.31 | 1103.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1117.40 | 1121.12 | 1113.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1117.40 | 1121.12 | 1113.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1113.20 | 1120.88 | 1113.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 1113.20 | 1120.88 | 1113.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1127.80 | 1120.95 | 1113.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 1130.20 | 1121.04 | 1113.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 1130.50 | 1121.25 | 1113.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 1129.70 | 1121.43 | 1113.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:00:00 | 1130.20 | 1121.81 | 1114.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1112.10 | 1122.45 | 1115.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 1112.00 | 1122.45 | 1115.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1108.80 | 1122.32 | 1115.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1108.80 | 1122.32 | 1115.11 | SL hit (close<static) qty=1.00 sl=1110.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 999.00 | 1108.17 | 1108.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 951.50 | 1104.65 | 1106.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 952.75 | 937.27 | 993.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 952.75 | 937.27 | 993.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 989.50 | 940.13 | 992.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 993.55 | 940.13 | 992.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 986.00 | 941.46 | 992.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:45:00 | 984.00 | 941.88 | 991.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 982.00 | 942.29 | 991.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1028.50 | 943.58 | 992.09 | SL hit (close>static) qty=1.00 sl=992.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-30 14:00:00 | 1325.05 | 2024-12-31 14:15:00 | 1364.15 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-12-31 09:15:00 | 1324.05 | 2024-12-31 14:15:00 | 1364.15 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-01-09 13:30:00 | 1322.75 | 2025-01-10 11:15:00 | 1256.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 13:30:00 | 1322.75 | 2025-01-13 12:15:00 | 1190.48 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-06-23 14:30:00 | 1249.10 | 2025-07-22 10:15:00 | 1224.10 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest1 | 2025-06-24 11:45:00 | 1252.50 | 2025-07-22 10:15:00 | 1224.10 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest1 | 2025-06-25 14:00:00 | 1254.70 | 2025-07-22 10:15:00 | 1224.10 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest1 | 2025-06-26 10:00:00 | 1250.30 | 2025-07-22 10:15:00 | 1224.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-09-16 12:15:00 | 1136.70 | 2025-09-25 09:15:00 | 1079.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 13:00:00 | 1134.00 | 2025-09-25 09:15:00 | 1077.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 1137.10 | 2025-09-25 09:15:00 | 1080.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 14:00:00 | 1138.80 | 2025-09-25 09:15:00 | 1081.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:00:00 | 1138.30 | 2025-09-25 09:15:00 | 1081.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:15:00 | 1136.70 | 2025-10-07 10:15:00 | 1023.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-16 13:00:00 | 1134.00 | 2025-10-07 10:15:00 | 1023.39 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2025-09-17 09:45:00 | 1137.10 | 2025-10-07 10:15:00 | 1024.92 | TARGET_HIT | 0.50 | 9.87% |
| SELL | retest2 | 2025-09-17 14:00:00 | 1138.80 | 2025-10-07 10:15:00 | 1024.47 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2025-09-19 12:00:00 | 1138.30 | 2025-10-07 11:15:00 | 1020.60 | TARGET_HIT | 0.50 | 10.34% |
| BUY | retest2 | 2025-12-09 10:45:00 | 1113.50 | 2026-01-07 10:15:00 | 1108.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-12-10 15:00:00 | 1111.40 | 2026-01-07 10:15:00 | 1108.80 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-12-11 10:30:00 | 1113.40 | 2026-01-07 10:15:00 | 1108.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-11 13:30:00 | 1111.30 | 2026-01-07 10:15:00 | 1108.80 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-01-01 09:45:00 | 1130.20 | 2026-01-07 13:15:00 | 1099.20 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-01-01 12:30:00 | 1130.50 | 2026-01-07 13:15:00 | 1099.20 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-01-01 15:00:00 | 1129.70 | 2026-01-07 13:15:00 | 1099.20 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-01-02 14:00:00 | 1130.20 | 2026-01-07 13:15:00 | 1099.20 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-02-12 13:45:00 | 984.00 | 2026-02-13 09:15:00 | 1028.50 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2026-02-12 14:30:00 | 982.00 | 2026-02-13 09:15:00 | 1028.50 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2026-02-23 09:45:00 | 973.70 | 2026-02-24 14:15:00 | 930.72 | PARTIAL | 0.50 | 4.41% |
| SELL | retest2 | 2026-02-23 12:00:00 | 979.70 | 2026-02-25 09:15:00 | 925.01 | PARTIAL | 0.50 | 5.58% |
| SELL | retest2 | 2026-02-23 09:45:00 | 973.70 | 2026-02-27 13:15:00 | 973.40 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest2 | 2026-02-23 12:00:00 | 979.70 | 2026-02-27 13:15:00 | 973.40 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2026-03-02 09:15:00 | 960.00 | 2026-03-04 11:15:00 | 912.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 960.00 | 2026-03-09 10:15:00 | 864.00 | TARGET_HIT | 0.50 | 10.00% |
